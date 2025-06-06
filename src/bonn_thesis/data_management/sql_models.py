"""Define SQLAlchemy ORM models that represent your database tables."""

import datetime
import decimal
from typing import Optional

from sqlalchemy import (
    Boolean,
    Date,
    Double,
    ForeignKeyConstraint,
    Index,
    Integer,
    Numeric,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "company"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="company_pkey"),
        UniqueConstraint("url", name="company_url_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(512))
    url: Mapped[str] = mapped_column(Text)
    call_to_action_url: Mapped[str | None] = mapped_column(Text)
    external_company_url: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(String(512))
    company_type: Mapped[str | None] = mapped_column(String(512))
    top_400: Mapped[bool | None] = mapped_column(Boolean)
    min_size: Mapped[int | None] = mapped_column(Integer)
    max_size: Mapped[int | None] = mapped_column(Integer)
    total_size: Mapped[int | None] = mapped_column(Integer)
    description: Mapped[str | None] = mapped_column(Text)
    tagline: Mapped[str | None] = mapped_column(String(512))
    founded: Mapped[int | None] = mapped_column(Integer)
    phone: Mapped[str | None] = mapped_column(String(512))
    followers_on_linkedin: Mapped[int | None] = mapped_column(Integer)

    company_industry: Mapped[list["CompanyIndustry"]] = relationship(
        "CompanyIndustry", back_populates="company"
    )
    company_location: Mapped[list["CompanyLocation"]] = relationship(
        "CompanyLocation", back_populates="company"
    )
    profile_experience: Mapped[list["ProfileExperience"]] = relationship(
        "ProfileExperience", back_populates="company"
    )


class Hierarchy(Base):
    __tablename__ = "hierarchy"
    __table_args__ = (PrimaryKeyConstraint("id", name="hierarchy_pkey"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(512))
    score: Mapped[int] = mapped_column(Integer)

    job_title_classification: Mapped[list["JobTitleClassification"]] = relationship(
        "JobTitleClassification", back_populates="hierarchy"
    )
    profile_experience: Mapped[list["ProfileExperience"]] = relationship(
        "ProfileExperience", back_populates="hierarchy"
    )


class Industry(Base):
    __tablename__ = "industry"
    __table_args__ = (
        PrimaryKeyConstraint("id", name="industry_pkey"),
        Index("ix_industry_industry", "industry", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    industry: Mapped[str] = mapped_column(String(512))

    company_industry: Mapped[list["CompanyIndustry"]] = relationship(
        "CompanyIndustry", back_populates="industry"
    )


class Params(Base):
    __tablename__ = "params"
    __table_args__ = (PrimaryKeyConstraint("id", name="params_pkey"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_education_score_avg: Mapped[float | None] = mapped_column(Double(53))
    company_education_score_std: Mapped[float | None] = mapped_column(Double(53))
    employee_cutoff: Mapped[int | None] = mapped_column(Integer)
    weighted_by_employees_duration: Mapped[bool | None] = mapped_column(Boolean)
    modification_date: Mapped[datetime.date | None] = mapped_column(Date)


class Profile(Base):
    __tablename__ = "profile"
    __table_args__ = (PrimaryKeyConstraint("id", name="profile_pkey"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str | None] = mapped_column(String(512))
    url: Mapped[str | None] = mapped_column(String(512))
    headline: Mapped[str | None] = mapped_column(String(512))
    gender: Mapped[str | None] = mapped_column(String(2))
    influencer: Mapped[bool | None] = mapped_column(Boolean)
    industry: Mapped[str | None] = mapped_column(String(512))
    skills: Mapped[str | None] = mapped_column(String(512))
    location: Mapped[str | None] = mapped_column(String(512))
    state: Mapped[str | None] = mapped_column(String(512))
    city: Mapped[str | None] = mapped_column(String(512))
    matched_city: Mapped[str | None] = mapped_column(String(512))
    country: Mapped[str | None] = mapped_column(String(512))
    latitude: Mapped[float | None] = mapped_column(Double(53))
    longitude: Mapped[float | None] = mapped_column(Double(53))
    crawling_date: Mapped[datetime.date | None] = mapped_column(Date)
    lang_locale_primary: Mapped[str | None] = mapped_column(String(512))
    lang_locale_supported_1: Mapped[str | None] = mapped_column(String(512))
    lang_locale_supported_2: Mapped[str | None] = mapped_column(String(512))
    lang_locale_supported_3: Mapped[str | None] = mapped_column(String(512))
    lang_profile_1: Mapped[str | None] = mapped_column(String(512))
    lang_profile_2: Mapped[str | None] = mapped_column(String(512))
    lang_profile_3: Mapped[str | None] = mapped_column(String(512))
    source: Mapped[str | None] = mapped_column(String(512))

    profile_education: Mapped[list["ProfileEducation"]] = relationship(
        "ProfileEducation", back_populates="profile"
    )
    profile_experience: Mapped[list["ProfileExperience"]] = relationship(
        "ProfileExperience", back_populates="profile"
    )


class CompanyIndustry(Base):
    __tablename__ = "company_industry"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id"], ["company.id"], name="company_industry_company_id_fkey"
        ),
        ForeignKeyConstraint(
            ["industry_id"], ["industry.id"], name="company_industry_industry_id_fkey"
        ),
        PrimaryKeyConstraint("id", name="company_industry_pkey"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer)
    industry_id: Mapped[int] = mapped_column(Integer)

    company: Mapped["Company"] = relationship(
        "Company", back_populates="company_industry"
    )
    industry: Mapped["Industry"] = relationship(
        "Industry", back_populates="company_industry"
    )


class CompanyLocation(Base):
    __tablename__ = "company_location"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id"],
            ["company.id"],
            ondelete="CASCADE",
            name="company_location_company_id_fkey",
        ),
        PrimaryKeyConstraint("id", name="company_location_pkey"),
        Index("ix_company_location_address", "address"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[int] = mapped_column(Integer)
    is_headquarter: Mapped[bool] = mapped_column(Boolean)
    country: Mapped[str | None] = mapped_column(String(2))
    geographic_area: Mapped[str | None] = mapped_column(String(512))
    city: Mapped[str | None] = mapped_column(String(512))
    postal_code: Mapped[str | None] = mapped_column(String(512))
    address: Mapped[str | None] = mapped_column(String(512))

    company: Mapped["Company"] = relationship(
        "Company", back_populates="company_location"
    )


class JobTitleClassification(Base):
    __tablename__ = "job_title_classification"
    __table_args__ = (
        ForeignKeyConstraint(
            ["hierarchy_id"],
            ["hierarchy.id"],
            name="job_title_classification_hierarchy_id_fkey",
        ),
        PrimaryKeyConstraint("id", name="job_title_classification_pkey"),
        Index("ix_job_title_classification_job_title", "job_title", unique=True),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_title: Mapped[str] = mapped_column(String(512))
    hierarchy_id: Mapped[int] = mapped_column(Integer)
    probability: Mapped[float | None] = mapped_column(Double(53))

    hierarchy: Mapped["Hierarchy"] = relationship(
        "Hierarchy", back_populates="job_title_classification"
    )
    profile_experience: Mapped[list["ProfileExperience"]] = relationship(
        "ProfileExperience", back_populates="job_title_classification"
    )


class ProfileEducation(Base):
    __tablename__ = "profile_education"
    __table_args__ = (
        ForeignKeyConstraint(
            ["profile_id"], ["profile.id"], name="profile_education_profile_id_fkey"
        ),
        PrimaryKeyConstraint("id", name="profile_education_pkey"),
        Index("idx_degree_type", "degree_type"),
        Index("idx_end_date", "end_date"),
        Index(
            "idx_id_end_date_type_subject_university",
            "id",
            "end_date",
            "degree_type",
            "subject",
            "university",
        ),
        Index("idx_potentially_scorable", "potentially_scorable"),
        Index("idx_profile_id", "profile_id"),
        Index("idx_subject", "subject"),
        Index("idx_university", "university"),
        Index("ix_profile_education_linkedin_url", "linkedin_url"),
        Index("ix_profile_education_pct_case", "pct_case"),
        Index("ix_profile_education_profile_id", "profile_id"),
        Index("ix_profile_education_university", "university"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(Integer)
    country: Mapped[str | None] = mapped_column(String(2))
    subject: Mapped[str | None] = mapped_column(String(512))
    grade: Mapped[str | None] = mapped_column(String(512))
    degree_type: Mapped[str | None] = mapped_column(String(512))
    start_date: Mapped[datetime.date | None] = mapped_column(Date)
    end_date: Mapped[datetime.date | None] = mapped_column(Date)
    location: Mapped[str | None] = mapped_column(String(512))
    university: Mapped[str | None] = mapped_column(String(512))
    linkedin_url: Mapped[str | None] = mapped_column(Text)
    details: Mapped[str | None] = mapped_column(String(512))
    case_degree_id: Mapped[int | None] = mapped_column(Integer)
    case_degree_label: Mapped[str | None] = mapped_column(String(512))
    case_matched_year: Mapped[int | None] = mapped_column(Integer)
    case_university: Mapped[str | None] = mapped_column(String(512))
    case_successfully_processed: Mapped[bool | None] = mapped_column(Boolean)
    case_not_enough_info: Mapped[bool | None] = mapped_column(Boolean)
    case_reason_for_failed_matching: Mapped[str | None] = mapped_column(String(512))
    pct_case: Mapped[float | None] = mapped_column(Double(53))
    pct_subject: Mapped[float | None] = mapped_column(Double(53))
    pct_local: Mapped[float | None] = mapped_column(Double(53))
    potentially_scorable: Mapped[bool | None] = mapped_column(Boolean)

    profile: Mapped["Profile"] = relationship(
        "Profile", back_populates="profile_education"
    )
    profile_education_scores: Mapped[list["ProfileEducationScores"]] = relationship(
        "ProfileEducationScores", back_populates="profile_education"
    )


class ProfileEducationScores(Base):
    __tablename__ = "profile_education_scores"
    __table_args__ = (
        ForeignKeyConstraint(
            ["profile_education_id"],
            ["profile_education.id"],
            name="profile_education_scores_profile_education_id_fkey",
        ),
        PrimaryKeyConstraint("id", name="profile_education_scores_pkey"),
        Index("idx_data_version", "data_version"),
        Index("idx_profile_education_id", "profile_education_id"),
        Index(
            "idx_profile_education_id_data_version",
            "profile_education_id",
            "data_version",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_education_id: Mapped[int] = mapped_column(Integer)
    data_version: Mapped[str] = mapped_column(
        String(255), server_default=text("'initial'::character varying")
    )
    pct_case: Mapped[decimal.Decimal | None] = mapped_column(Numeric(5, 2))
    pct_global: Mapped[decimal.Decimal | None] = mapped_column(Numeric(5, 2))
    case_degree_id: Mapped[int | None] = mapped_column(Integer)
    case_degree_label: Mapped[str | None] = mapped_column(String(255))
    case_matched_year: Mapped[int | None] = mapped_column(Integer)
    case_university: Mapped[str | None] = mapped_column(String(255))
    case_successfully_processed: Mapped[bool | None] = mapped_column(Boolean)
    case_not_enough_info: Mapped[bool | None] = mapped_column(Boolean)
    case_reason_for_failed_matching: Mapped[str | None] = mapped_column(Text)

    profile_education: Mapped["ProfileEducation"] = relationship(
        "ProfileEducation", back_populates="profile_education_scores"
    )


class ProfileExperience(Base):
    __tablename__ = "profile_experience"
    __table_args__ = (
        ForeignKeyConstraint(
            ["company_id"], ["company.id"], name="profile_experience_company_id_fkey"
        ),
        ForeignKeyConstraint(
            ["hierarchy_id"],
            ["hierarchy.id"],
            name="profile_experience_hierarchy_id_fkey",
        ),
        ForeignKeyConstraint(
            ["job_title_classification_id"],
            ["job_title_classification.id"],
            name="profile_experience_job_title_classification_id_fkey",
        ),
        ForeignKeyConstraint(
            ["profile_id"], ["profile.id"], name="profile_experience_profile_id_fkey"
        ),
        PrimaryKeyConstraint("id", name="profile_experience_pkey"),
        Index("ix_profile_experience_company_id", "company_id"),
        Index("ix_profile_experience_duration", "duration"),
        Index("ix_profile_experience_hierarchy_id", "hierarchy_id"),
        Index("ix_profile_experience_job_title_cleaned", "job_title_cleaned"),
        Index("ix_profile_experience_profile_id", "profile_id"),
        Index("ix_profile_experience_total_experience", "total_experience"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(Integer)
    job_title: Mapped[str | None] = mapped_column(String(512))
    total_experience: Mapped[float | None] = mapped_column(Double(53))
    gender: Mapped[str | None] = mapped_column(String(2))
    company_name: Mapped[str | None] = mapped_column(String(512))
    job_title_cleaned: Mapped[str | None] = mapped_column(String(512))
    description: Mapped[str | None] = mapped_column(Text)
    duration: Mapped[float | None] = mapped_column(Double(53))
    start_date: Mapped[datetime.date | None] = mapped_column(Date)
    end_date: Mapped[datetime.date | None] = mapped_column(Date)
    present: Mapped[bool | None] = mapped_column(Boolean)
    location: Mapped[str | None] = mapped_column(String(512))
    score: Mapped[float | None] = mapped_column(Double(53))
    company_rank: Mapped[float | None] = mapped_column(Double(53))
    is_last_experience: Mapped[bool | None] = mapped_column(Boolean)
    company_id: Mapped[int | None] = mapped_column(Integer)
    job_title_classification_id: Mapped[int | None] = mapped_column(Integer)
    hierarchy_id: Mapped[int | None] = mapped_column(Integer)

    company: Mapped[Optional["Company"]] = relationship(
        "Company", back_populates="profile_experience"
    )
    hierarchy: Mapped[Optional["Hierarchy"]] = relationship(
        "Hierarchy", back_populates="profile_experience"
    )
    job_title_classification: Mapped[Optional["JobTitleClassification"]] = relationship(
        "JobTitleClassification", back_populates="profile_experience"
    )
    profile: Mapped["Profile"] = relationship(
        "Profile", back_populates="profile_experience"
    )
