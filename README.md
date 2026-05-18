## Project Overview

The chatbot is provided structured and unstructred data. The chatbot is only allowed to answer questions based on the data provided to it.

The structured data is provided through a databricks SQL database titled "chatbot". All tables in the schema are gathered from Kaggle, the USDA Food Safety and Inspection Service, and the Centers for Disease Control and Prevention. The tables/columns provided are as such:

air_pollution_and_health_risk_death_rates:
Year
Nitrogen_Oxide_emmision_in_tons
Sulphur_Dioxide_emmision_in_tons
Carbon_Monoxide_emmision_in_tons
Organic_Carbon_emmision_in_tons
NMVOCs_emmision_in_tons
Black_Carbon_emmision_in_tons
Ammonia_emmision_in_tons
avg_PM2_5_concentration
Avg_Status
Avg_AQI_Value
Deaths_Air_pollution_Sex_Both_Age_Age_standardized_Rate
Deaths_Household_air_pollution_from_solid_fuels_Sex_Both_Age_Age_standardized_Rate
Deaths_Ambient_particulate_matter_pollution_Sex_Both_Age_Age_standardized_Rate
Deaths_Ambient_ozone_pollution_Sex_Both_Age_Age_standardized_Rate
Deaths_by_Unsafe_water_source
Deaths_by_Unsafe_sanitation
Deaths_by_No_access_to_handwashing_facility
Deaths_by_Household_air_pollution_from_solid_fuels
Deaths_by_Non_exclusive_breastfeeding
Deaths_by_Discontinued_breastfeeding
Deaths_by_Child_wasting
Deaths_by_Child_stunting
Deaths_by_Low_birth_weight_for_gestation
Deaths_by_Secondhand_smoke
Deaths_by_Alcohol_use
Deaths_by_Drug_use
Deaths_by_Diet_low_in_fruits
Deaths_by_Diet_low_in_vegetables
Deaths_by_Unsafe_sex
Deaths_by_Low_physical_activity
Deaths_by_High_fasting_plasma_glucose
Deaths_by_High_total_cholesterol
Deaths_by_High_body_mass_index
Deaths_by_High_systolic_blood_pressure
Deaths_by_Smoking
Deaths_by_Iron_deficiency
Deaths_by_Vitamin_A_deficiency
Deaths_by_Low_bone_mineral_density
Deaths_by_Air_pollution
Deaths_by_Outdoor_air_pollution
Deaths_by_Diet_high_in_sodium
Deaths_by_Diet_low_in_whole_grains
Deaths_by_Diet_low_in_nuts_and_seeds
Deaths_by_Air_pollution_total_IHME_2019
CO2_emissions_metric_million_tons

cancer_death_rates_in_the_us:
Year
Breast_female
Colon_and_Rectum_female
Colon_and_Rectum_male
Leukemia_male
Liver_female
Liver_male
Lung_and_Bronchus_female
Lung_and_Bronchus_male
Pancreas_female
Pancreas_male
Prostate_male
Stomach_female
Stomach_male
Uterus_female

nhanes_select_chronic_conditions_prevalence_estimates:
Survey_Years
Sex
Age_Group
Race_and_Hispanic_Origin
Measure
Percent
Standard_Error
Lower_95_CI_Limit
Upper_95_CI_Limit
Presentation_Standard
Note1
Notea

import_presented_refused_fy_202(2-6): (USDA FSIS records of imported food products that were presented for inspection and refused entry)
lot_id
application_number
lot_number
country
received_lot_date
refused_date
disposition_date
import_house
import_house_name
processing_establishment
processing_est_name
haccp_code
process_category
product_category
product_group
species
total_weight_presented
final_weight_refused

import_refusal_reason_fy_202(2-6):
lot_id
refusal_create_date
refusal_reason
defect_description
failed_toi
refused_weight
rectify_status

The table "air_pollution_and_health_risk_death_rates" was derived by joining and aggregating data from multiple Kaggle datasets. 

Unstructured data is provided through vectorized pdf documents. The pdfs are given in /pdfs, with sentence-transformers used to locally vectorize pdfs and FAISS used for vector search. The pdfs used are:

Baseline_Data_Raw_Chicken_Parts.pdf
Baseline_Data_Young_Turkey_2008-2009.pdf
cystic_fibrosis_research.pdf
State-of-the-Air-2025.pdf
Beef-Veal-Carcass-Baseline-Study-Report.pdf
Baseline-Raw-Liquid-Eggs_0.pdf
ghe2021_cod_methods.pdf
salmonella-fact-sheet-2015.pdf
Baseline_Data_Market_Hogs_2010-2011.pdf
Baseline_Data_Young_Chicken_2007-2008.pdf

The project requires Databricks credentials and an OpenAI API key to run (see: .env.example)

I am currently in the process of transitioning my frontend to LibreChat UI (https://github.com/danny-avila/LibreChat). In app.py, the commented block at the bottom of the file is the LibreChat integration code, which currently only supports text generated messages. When using LibreChat, this block replaces the section from "@app.route("/")" to "app.run()".

## Planned Additions

- Transitioning frontend to LibreChat UI
- Improving the formatting of the AI generated responses
- Migration from OpenAI to Ollama

## Architecture

User questions are routed through a two-stage decision pipeline:

**Stage 1 — SQL or PDF**
An LLM classifies the user's question as requiring either structured or unstructured data:

Questions about numbers, trends, or statistics are routed to SQL, where a second LLM generates and executes a SELECT query

Conceptual or research-based questions are routed to the vectorized PDFs, where the question is embedded locally and the most similar chunks are retrieved via FAISS

**Stage 2 — TEXT, TABLE, or CHART**
A separate LLM call decides what format to return the answer in based on the user's question. The supported formats are text-based, a table, or a chart. 

SQL queries are validated before execution to ensure the LLM does not modify the tables in the databricks schema. User inputs are also screened by a guardrail check before any LLM calls are made.


## Tech Stack

Backend: Python, Flask
Frontend: HTML, CSS, JavaScript, Chart.js
LLM: OpenAI GPT-4o-mini
Embeddings: sentence-transformers
Vector Search: FAISS
Database: Databricks SQL

Signed: Nick Myers