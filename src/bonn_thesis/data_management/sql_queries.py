"""SQL queries for data extraction."""

import logging

import pandas as pd

from bonn_thesis.data_management.sql_connection import db_connection

logger = logging.getLogger(__name__)


def extract_profile(limit: int | None = 10000, offset: int = 0) -> pd.DataFrame:
    """Extract profiles from the database.

    Args:
        limit: Number of profiles to extract
        offset: Starting offset for pagination (in terms of profiles)

    Returns:
        DataFrame containing profile data
    """
    query = """
    SELECT
        id AS prof_id,
        gender,
        location AS prof_location,
        city AS prof_city,
        state AS prof_state,
        country AS prof_country,
        industry,
        crawling_date
    FROM profile
    ORDER BY id
    """

    if limit is not None:
        query += f" LIMIT {limit}"

    if offset > 0:
        query += f" OFFSET {offset}"

    with db_connection() as conn:
        logger.info("Extracting profiles (limit=%s, offset=%s)", limit, offset)
        return pd.read_sql_query(query, conn)


def extract_experience(limit: int | None = 10000, offset: int = 0) -> pd.DataFrame:
    """Extract profile experiences from the database.

    Args:
        limit: Number of profiles to extract experiences for
        offset: Starting offset for pagination (in terms of profiles)

    Returns:
        DataFrame containing experience data
    """
    # Step 1: Get distinct profile IDs with pagination
    with db_connection() as conn:
        profile_query = """
        SELECT id AS profile_id
        FROM profile
        ORDER BY id
        """

        if limit is not None:
            profile_query += f" LIMIT {limit}"

        if offset > 0:
            profile_query += f" OFFSET {offset}"

        profile_ids_df = pd.read_sql_query(profile_query, conn)

    if profile_ids_df.empty:
        logger.info("No profiles found with offset %s", offset)
        return pd.DataFrame()

    profile_ids = profile_ids_df["profile_id"].tolist()

    # Step 2: Get ALL experiences for these complete profiles
    if len(profile_ids) == 1:
        where_clause = "profile_id = %s"
        where_params = (profile_ids[0],)
    else:
        placeholders = ",".join(["%s"] * len(profile_ids))
        where_clause = f"profile_id IN ({placeholders})"
        where_params = tuple(profile_ids)

    query = (
        """
    SELECT
        id AS exp_id,
        profile_id AS prof_id,
        company_id AS comp_id,
        job_title_classification_id AS job_title_id,
        job_title,
        job_title_cleaned,
        description,
        company_name AS exp_company,
        total_experience,
        duration,
        start_date,
        end_date,
        present,
        is_last_experience,
        location AS exp_location,
        hierarchy_id AS hierarchy
    FROM
        profile_experience
    WHERE
        """
        + where_clause
        + """
    ORDER BY
        profile_id, id
    """
    )

    with db_connection() as conn:
        logger.info("Extracting experiences for %s profiles", len(profile_ids))
        return pd.read_sql_query(query, conn, params=where_params)


def extract_education(limit: int | None = 10000, offset: int = 0) -> pd.DataFrame:
    """Extract profile education records from the database.

    Args:
        limit: Number of profiles to extract education records for
        offset: Starting offset for pagination (in terms of profiles)

    Returns:
        DataFrame containing education data
    """
    # Step 1: Get distinct profile IDs with pagination
    with db_connection() as conn:
        profile_query = """
        SELECT id AS profile_id
        FROM profile
        ORDER BY id
        """

        if limit is not None:
            profile_query += f" LIMIT {limit}"

        if offset > 0:
            profile_query += f" OFFSET {offset}"

        profile_ids_df = pd.read_sql_query(profile_query, conn)

    if profile_ids_df.empty:
        logger.info("No profiles found with offset %s", offset)
        return pd.DataFrame()

    profile_ids = profile_ids_df["profile_id"].tolist()

    # Step 2: Get ALL education records for these complete profiles
    if len(profile_ids) == 1:
        where_clause = "profile_id = %s"
        where_params = (profile_ids[0],)
    else:
        placeholders = ",".join(["%s"] * len(profile_ids))
        where_clause = f"profile_id IN ({placeholders})"
        where_params = tuple(profile_ids)

    query = (
        """
    SELECT
        id AS ed_id,
        profile_id AS prof_id,
        country AS ed_country,
        subject,
        grade,
        degree_type,
        start_date,
        end_date,
        university,
        details
    FROM
        profile_education
    WHERE
        """
        + where_clause
        + """
    ORDER BY
        profile_id, id
    """
    )

    with db_connection() as conn:
        logger.info("Extracting education records for %s profiles", len(profile_ids))
        return pd.read_sql_query(query, conn, params=where_params)


def extract_companies(limit: int | None = 10000, offset: int = 0) -> pd.DataFrame:
    """Extract companies from the database.

    Args:
        limit: Number of companies to extract
        offset: Starting offset for pagination

    Returns:
        DataFrame containing company data
    """
    query = """
    SELECT
        id AS comp_id,
        name AS company,
        location AS comp_location,
        company_type AS comp_type,
        top_400,
        min_size,
        max_size,
        total_size,
        founded,
        followers_on_linkedin
    FROM
        company
    ORDER BY
        id
    """

    if limit is not None:
        query += f" LIMIT {limit}"

    if offset > 0:
        query += f" OFFSET {offset}"

    with db_connection() as conn:
        logger.info("Extracting companies (limit=%s, offset=%s)", limit, offset)
        return pd.read_sql_query(query, conn)


def extract_company_location() -> pd.DataFrame:
    """Extract company locations."""
    query = """
    SELECT
        company_id AS comp_id,
        city AS comp_city,
        postal_code AS comp_postal_code,
        address AS comp_address,
        is_headquarter AS comp_headquarter
    FROM
        company_location
    ORDER BY
        company_id
    """

    with db_connection() as conn:
        return pd.read_sql_query(query, conn)


def extract_industry_classifications() -> pd.DataFrame:
    """Extract industry classifications."""
    query = """
    SELECT
        id AS industry_id,
        industry
    FROM
        industry
    ORDER BY
        id
    """

    with db_connection() as conn:
        return pd.read_sql_query(query, conn)


def extract_merged_linkedin_data(
    limit: int | None = 10000, offset: int = 0
) -> pd.DataFrame:
    """Extract merged LinkedIn data by complete profiles.

    This ensures that all experiences from a profile are included together,
    avoiding split profiles across different batches.

    Args:
        limit: Number of profiles to extract per batch
        offset: Starting offset for pagination (in terms of profiles)

    Returns:
        DataFrame containing comprehensive joined LinkedIn data
    """
    # Step 1: Get distinct profile IDs with pagination
    with db_connection() as conn:
        profile_query = """
        SELECT id AS profile_id
        FROM profile
        ORDER BY id
        """

        if limit is not None:
            profile_query += f" LIMIT {limit}"

        if offset > 0:
            profile_query += f" OFFSET {offset}"

        profile_ids_df = pd.read_sql_query(profile_query, conn)

    if profile_ids_df.empty:
        logger.info("No profiles found with offset %s", offset)
        return pd.DataFrame()

    profile_ids = profile_ids_df["profile_id"].tolist()
    logger.info("Selected %s complete profiles", len(profile_ids))

    # Step 2: Get ALL experiences for these complete profiles
    # Handle single profile case
    if len(profile_ids) == 1:
        where_clause = "pe.profile_id = %s"
        where_params = (profile_ids[0],)
    else:
        # Create placeholders for each profile ID
        placeholders = ",".join(["%s"] * len(profile_ids))
        where_clause = f"pe.profile_id IN ({placeholders})"
        where_params = tuple(profile_ids)

    # Step 3: Get full data for all experiences from these profiles
    batch_query = (
        """
        SELECT
            -- Profile Experience Data
            pe.id AS exp_id,
            pe.profile_id AS prof_id,
            pe.company_id AS comp_id,
            pe.job_title_classification_id AS job_title_id,
            pe.job_title,
            pe.job_title_cleaned,
            pe.description AS exp_description,
            pe.company_name AS exp_company,
            pe.total_experience,
            pe.duration,
            pe.start_date AS exp_start_date,
            pe.end_date AS exp_end_date,
            pe.present,
            pe.is_last_experience,
            pe.location AS exp_location,
            pe.hierarchy_id AS hierarchy,

            -- Profile Data
            p.gender,
            p.location AS prof_location,
            p.city AS prof_city,
            p.state AS prof_state,
            p.country AS prof_country,
            p.industry AS prof_industry,
            p.crawling_date,

            -- Company Data
            c.name AS company,
            c.location AS comp_location,
            c.company_type AS comp_type,
            c.top_400,
            c.min_size,
            c.max_size,
            c.total_size,
            c.founded,
            c.followers_on_linkedin,

            -- Company Location (headquarters)
            cl.city AS comp_city,
            cl.postal_code AS comp_postal_code,
            cl.address AS comp_address,
            cl.is_headquarter AS comp_headquarter,

            -- Industry Data (via company_industry)
            i.id AS industry_id,
            i.industry,

            -- Hierarchy Data
            h.name AS hierarchy_name,

            -- Job Title Classification
            jtc.job_title AS job_title_standard

        FROM
            profile_experience pe

        -- Join with profile
        LEFT JOIN
            profile p ON pe.profile_id = p.id
        -- Join with company
        LEFT JOIN
            company c ON pe.company_id = c.id
        -- Join with company_location (for headquarters)
        LEFT JOIN
            company_location cl ON c.id = cl.company_id AND cl.is_headquarter = true
        -- Join with company_industry and industry
        LEFT JOIN
            company_industry ci ON c.id = ci.company_id
        LEFT JOIN
            industry i ON ci.industry_id = i.id
        -- Join with hierarchy
        LEFT JOIN
            hierarchy h ON pe.hierarchy_id = h.id
        -- Join with job_title_classification
        LEFT JOIN
            job_title_classification jtc ON pe.job_title_classification_id = jtc.id

        WHERE
            """
        + where_clause
        + """
        ORDER BY
            pe.profile_id, pe.id
        """
    )

    with db_connection() as conn:
        result_df = pd.read_sql_query(batch_query, conn, params=where_params)

        logger.info(
            "Retrieved %s experiences from %s complete profiles",
            len(result_df),
            result_df["prof_id"].nunique() if not result_df.empty else 0,
        )

        return result_df


def _get_profile_count() -> int:
    """Get total count of profiles in the database."""
    with db_connection() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM profile")
        return cursor.fetchone()[0]


def _get_experience_count() -> int:
    """Get total count of experiences in the database."""
    with db_connection() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM profile_experience")
        return cursor.fetchone()[0]


def _get_education_count() -> int:
    """Get total count of education records in the database."""
    with db_connection() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM profile_education")
        return cursor.fetchone()[0]


def _get_company_count() -> int:
    """Get total count of companies in the database."""
    with db_connection() as conn, conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM company")
        return cursor.fetchone()[0]
