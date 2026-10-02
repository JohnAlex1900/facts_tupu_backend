import os
import json
import asyncio
import logging
import time
import asyncpg
from dotenv import load_dotenv

from ai_service import generate_true_ai_deep_dive

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("SocialSyncWorker")

DATABASE_URL = os.getenv("SUPABASE_DATABASE_URL")


async def ensure_db_schema(pool: asyncpg.Pool):
    """Ensures the cache table for AI reports exists with appropriate constraints."""
    async with pool.acquire() as conn:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS representative_ai_reports (
                representative_id VARCHAR(255) PRIMARY KEY,
                ai_report_data JSONB NOT NULL,
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        logger.info("Database schema verified (representative_ai_reports table confirmed).")


async def process_representative(
    pool: asyncpg.Pool, 
    sem: asyncio.Semaphore, 
    rep: dict, 
    index: int, 
    total_count: int,
    stats: dict
):
    async with sem:
        rep_id = rep["id"]
        rep_name = rep["name"]
        
        try:
            logger.info(f"[{index}/{total_count}] Processing AI report for [{rep['role']}]: {rep_name} ({rep['county']})...")
            
            ai_json_report = await generate_true_ai_deep_dive(
                name=rep_name,
                role=rep["role"],
                county=rep["county"],
                jaba=float(rep["jaba_meter"]),
                impact=float(rep["impact_rating"]),
                rvs=float(rep["rvs"])
            )
            
            async with pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO representative_ai_reports (
                        representative_id, ai_report_data, last_updated
                    )
                    VALUES ($1, $2, CURRENT_TIMESTAMP)
                    ON CONFLICT (representative_id) 
                    DO UPDATE SET 
                        ai_report_data = EXCLUDED.ai_report_data,
                        last_updated = CURRENT_TIMESTAMP;
                """, rep_id, json.dumps(ai_json_report))
                
            stats["success"] += 1
            logger.info(f"[{index}/{total_count}] Successfully synced {rep_name} | Overall Score: {ai_json_report.get('overall_score', 0):.1f}/100")
            
        except Exception as e:
            stats["failed"] += 1
            logger.error(f"[{index}/{total_count}] Failed processing profile for {rep_name} (ID: {rep_id}): {str(e)}")


async def sync_all_representatives():
    """
    Runs the daily pipeline: fetches all active Kenyan leaders across all administrative layers,
    queries web context + Groq, and caches structured constitutional reports into PostgreSQL/Supabase.
    """
    start_time = time.time()
    
    if not DATABASE_URL:
        logger.error("FATAL: SUPABASE_DATABASE_URL missing from environment variables.")
        return

    logger.info("Connecting to database pool...")
    pool = await asyncpg.create_pool(dsn=DATABASE_URL, min_size=1, max_size=10)
    
    try:
        # Guarantee schema setup
        await ensure_db_schema(pool)

        # Primary selection query aligning exactly with main.py ID patterns
        unified_rep_query = """
            -- 1. Governors
            SELECT 
                'inc-governor-' || co.county_code AS id, 
                ces.governor_name AS name, 
                'County Governor' AS role, 
                co.county_name AS county,
                COALESCE(m.jaba_meter, 15) AS jaba_meter,
                COALESCE(m.performance_score, 70) AS impact_rating,
                COALESCE(m.risk_radar_index, 10) AS rvs
            FROM administrative_counties co
            JOIN county_executive_senate ces ON co.county_code = ces.county_code
            LEFT JOIN incumbent_accountability_metrics m ON m.target_type = 'governor' AND m.associated_id = co.county_code
            WHERE ces.governor_name IS NOT NULL AND TRIM(ces.governor_name) != ''
            
            UNION ALL

            -- 2. Senators
            SELECT 
                'inc-senator-' || co.county_code AS id, 
                ces.senator_name AS name, 
                'County Senator' AS role, 
                co.county_name AS county,
                COALESCE(m.jaba_meter, 15) AS jaba_meter,
                COALESCE(m.performance_score, 70) AS impact_rating,
                COALESCE(m.risk_radar_index, 10) AS rvs
            FROM administrative_counties co
            JOIN county_executive_senate ces ON co.county_code = ces.county_code
            LEFT JOIN incumbent_accountability_metrics m ON m.target_type = 'senator' AND m.associated_id = co.county_code
            WHERE ces.senator_name IS NOT NULL AND TRIM(ces.senator_name) != ''

            UNION ALL

            -- 3. Woman Representatives (ID aligned with main.py 'inc-women_rep-')
            SELECT 
                'inc-women_rep-' || co.county_code AS id, 
                ces.women_rep_name AS name, 
                'Woman Representative' AS role, 
                co.county_name AS county,
                COALESCE(m.jaba_meter, 15) AS jaba_meter,
                COALESCE(m.performance_score, 70) AS impact_rating,
                COALESCE(m.risk_radar_index, 10) AS rvs
            FROM administrative_counties co
            JOIN county_executive_senate ces ON co.county_code = ces.county_code
            LEFT JOIN incumbent_accountability_metrics m ON m.target_type = 'women_rep' AND m.associated_id = co.county_code
            WHERE ces.women_rep_name IS NOT NULL AND TRIM(ces.women_rep_name) != ''

            UNION ALL

            -- 4. National Executive
            SELECT 
                ne.office_id AS id, 
                ne.leader_name AS name, 
                ne.role AS role, 
                'Kenya' AS county,
                COALESCE(m.jaba_meter, 15) AS jaba_meter,
                COALESCE(m.performance_score, 70) AS impact_rating,
                COALESCE(m.risk_radar_index, 10) AS rvs
            FROM national_executive ne
            LEFT JOIN incumbent_accountability_metrics m 
                ON (m.target_type = 'president' OR m.target_type = 'deputy_president') 
                AND m.associated_id = ne.office_id
            WHERE (ne.status = 'Active' OR ne.status IS NULL) AND ne.leader_name IS NOT NULL AND TRIM(ne.leader_name) != ''

            UNION ALL

            -- 5. Members of Parliament (Constituencies)
            SELECT 
                'inc-mp-' || pc.constituency_id AS id,
                pc.mp_name AS name,
                'Member of Parliament' AS role,
                co.county_name AS county,
                COALESCE(m.jaba_meter, 15) AS jaba_meter,
                COALESCE(m.performance_score, 70) AS impact_rating,
                COALESCE(m.risk_radar_index, 10) AS rvs
            FROM parliament_constituencies pc
            JOIN administrative_counties co ON pc.county_code = co.county_code
            LEFT JOIN incumbent_accountability_metrics m ON m.target_type = 'mp' AND m.associated_id = CAST(pc.constituency_id AS text)
            WHERE pc.mp_name IS NOT NULL AND TRIM(pc.mp_name) != ''

            UNION ALL

            -- 6. Members of County Assembly (Wards)
            SELECT 
                'inc-mca-' || law.ward_id AS id,
                law.mca_name AS name,
                'Member of County Assembly' AS role,
                co.county_name AS county,
                COALESCE(m.jaba_meter, 15) AS jaba_meter,
                COALESCE(m.performance_score, 70) AS impact_rating,
                COALESCE(m.risk_radar_index, 10) AS rvs
            FROM local_assembly_wards law
            JOIN administrative_counties co ON law.county_code = co.county_code
            LEFT JOIN incumbent_accountability_metrics m ON m.target_type = 'mca' AND m.associated_id = CAST(law.ward_id AS text)
            WHERE law.mca_name IS NOT NULL AND TRIM(law.mca_name) != ''
        """
        
        async with pool.acquire() as conn:
            try:
                representatives = await conn.fetch(unified_rep_query)
            except Exception as query_err:
                logger.warning(f"Schema fallback triggered due to missing metrics tables: {query_err}")
                fallback_query = """
                    SELECT 
                        'inc-governor-' || co.county_code AS id, 
                        ces.governor_name AS name, 
                        'County Governor' AS role, 
                        co.county_name AS county,
                        15.0 AS jaba_meter, 70.0 AS impact_rating, 10.0 AS rvs
                    FROM administrative_counties co
                    JOIN county_executive_senate ces ON co.county_code = ces.county_code
                    WHERE ces.governor_name IS NOT NULL AND TRIM(ces.governor_name) != ''
                    
                    UNION ALL
    
                    SELECT 
                        'inc-senator-' || co.county_code AS id, 
                        ces.senator_name AS name, 
                        'County Senator' AS role, 
                        co.county_name AS county,
                        15.0 AS jaba_meter, 70.0 AS impact_rating, 10.0 AS rvs
                    FROM administrative_counties co
                    JOIN county_executive_senate ces ON co.county_code = ces.county_code
                    WHERE ces.senator_name IS NOT NULL AND TRIM(ces.senator_name) != ''
    
                    UNION ALL
    
                    SELECT 
                        'inc-women_rep-' || co.county_code AS id, 
                        ces.women_rep_name AS name, 
                        'Woman Representative' AS role, 
                        co.county_name AS county,
                        15.0 AS jaba_meter, 70.0 AS impact_rating, 10.0 AS rvs
                    FROM administrative_counties co
                    JOIN county_executive_senate ces ON co.county_code = ces.county_code
                    WHERE ces.women_rep_name IS NOT NULL AND TRIM(ces.women_rep_name) != ''
    
                    UNION ALL
    
                    SELECT 
                        ne.office_id AS id, 
                        ne.leader_name AS name, 
                        ne.role AS role, 
                        'Kenya' AS county,
                        15.0 AS jaba_meter, 70.0 AS impact_rating, 10.0 AS rvs
                    FROM national_executive ne
                    WHERE (ne.status = 'Active' OR ne.status IS NULL) AND ne.leader_name IS NOT NULL;
                """
                representatives = await conn.fetch(fallback_query)

        total_count = len(representatives)
        logger.info(f"Loaded {total_count} active representatives across all administrative layers.")

        # Concurrency limit of 3 stays within Groq rate limits smoothly
        sem = asyncio.Semaphore(3)
        stats = {"success": 0, "failed": 0}
        
        tasks = [
            process_representative(pool, sem, dict(rep), index, total_count, stats)
            for index, rep in enumerate(representatives, 1)
        ]
        
        await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"Daily AI Generation pipeline completed in {elapsed_time:.2f}s | "
            f"Success: {stats['success']} | Failed: {stats['failed']}"
        )
                
    finally:
        await pool.close()

if __name__ == "__main__":
    asyncio.run(sync_all_representatives())