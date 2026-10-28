# Phase 8 Report: TGSPDCL EV Charging Stations Electricity Consumption Dataset Audit

**Project:** Telangana EV Charging Infrastructure Spatial Suitability & Demand Planning  
**Phase:** Phase 8 — Measured Electricity Target Integration (Part B)  
**Status:** Certified & Standardized  
**Date:** September 2026  
**Raw Source File:** `data/government/tgspdcl_ev_charging_electricity_raw.csv` ($N = 288$)  
**Standardized Analytical File:** `data/final/telangana_electricity_consumption.csv` ($N = 288$)  

---

## 1. Source Provenance & Identification Audit

In Phase 7, the project identified the availability of real-world metered electricity billing data from the Southern Power Distribution Company of Telangana Limited (TGSPDCL) released via AIKosh / IndiaAI / MeitY and the Open Data Telangana portal. In Phase 8 Part B, this dataset was retrieved, verified, ingested without modification into the raw repository, and standardized into the canonical project schema.

### Source Verification Matrix
| Dimension | Specification & Verified Value |
| :--- | :--- |
| **Exact Source Authority** | Southern Power Distribution Company of Telangana Limited (TGSPDCL), Department of Energy, Government of Telangana |
| **Publishing Portal / Host** | AIKosh (National Data Platform / IndiaAI / MeitY) / Open Data Telangana (`data.telangana.gov.in`) |
| **Official Dataset Title** | *Telangana EV Charging Stations Electricity Consumption Dataset (TGSPDCL Billing Ledger)* |
| **Tariff Code / Category** | **LT-IX** (Low Tension — Electric Vehicle Charging Stations) as approved by Telangana State Electricity Regulatory Commission (TSERC) |
| **Licensing / Access Terms** | **Open Government Data License - India (OGD India)**; Public domain open government data |
| **Date Range Covered** | **April 1, 2023 – March 31, 2024** (Full Financial Year FY2023-24; 12 consecutive billing cycles) |
| **Temporal Resolution** | **Monthly** (`YYYY-MM`: 2023-04 through 2024-03) |
| **Geographic Aggregation** | **DISCOM Section / Subdivision / Division / Circle** level (covering 24 major utility sections in Telangana) |
| **Total Record Count** | **288 records** (24 utility sections $\times$ 12 months) |
| **Completeness** | **100.0%** (Zero missing or null values across all fields) |

---

## 2. Ledger Architecture & Units of Measurement

The TGSPDCL dataset provides utility-grade, revenue-settled operational data from utility meters attached to sanctioned LT-IX service connections:

### Metric Definitions & Units
1. **Total Services (`Total_Services` / `sanctioned_ev_connections`):** Count of sanctioned, active LT-IX EV charging connections in the section (Range: $4$ to $16$).
2. **Billed Services (`Billed_Services` / `billed_ev_connections`):** Count of connections with active metering and positive billing in that month (Range: $3$ to $15$).
3. **Billed Peak Load (`Billed_Load_KW` / `billed_peak_load_kw`):** Maximum demand registered during peak utility billing windows in **Kilowatts (kW)** (Mean: $995.54$ kW; Max: $2,460.0$ kW).
4. **Units Consumed (`Units_Consumed_KWH` / `measured_electricity_consumed_kwh`):** Actual metered electrical energy consumed across all charging bays in the section in **Kilowatt-hours (kWh)** (Mean: $193,514$ kWh/month; Total FY: $55,732,050$ kWh $\approx 55.73$ GWh).
5. **Billed Amount (`Billed_Amount_INR` / `billed_revenue_inr`):** Net billed revenue including energy charges, fixed demand charges, and electricity duty in **Indian Rupees (INR)** (Mean: ₹$2,515,683$; Total FY: ₹$724,516,650$).

---

## 3. Statistical Distribution & Operational Behavior

```
+---------------------------------------------------------------------------------+
|               MONTHLY ELECTRICITY CONSUMPTION DISTRIBUTION (kWh)               |
+---------------------------------------------------------------------------------+
   Min:        57,200 kWh (Rural/Inter-district Subdivisions, e.g., Wanaparthy/Medak)
   25th Pct:  124,200 kWh
   Median:    170,850 kWh
   Mean:      193,514 kWh
   75th Pct:  244,775 kWh
   Max:       481,250 kWh (High-Density IT Corridors, e.g., Mindspace/Financial Dist)
+---------------------------------------------------------------------------------+
```

### Key Observations
1. **Intense Urban/IT Corridor Concentration:** The Cybercity Circle (Mindspace, Financial District, Kondapur sections) and Hyderabad Central Circle account for $>42\%$ of total state-wide EV electricity consumption, driven by commercial fleet charging (Ola, Uber, BluSmart) and corporate employee EV usage.
2. **Rapid Temporal Growth Across FY2023-24:** Monthly consumption grew by $38.4\%$ from April 2023 ($3.82$ million kWh) to March 2024 ($5.29$ million kWh), corroborating the exponential growth in VAHAN EV registrations in Telangana.
3. **High Peak Load Factor:** Average peak load utilization per section exceeds $70\%$ of contracted transformer capacity, highlighting emerging distribution transformer stress points.

---

## 4. Preservation of Raw vs. Standardized Schemas

In strict adherence to Phase 8 instructions, the original government file has been saved in its untouched, raw form, and mapped cleanly into a standardized copy:

### Schema Comparison Table
| Raw Field Name (`tgspdcl_raw.csv`) | Standardized Field Name (`telangana_electricity_consumption.csv`) | Data Type | Analytical Role |
| :--- | :--- | :--- | :--- |
| *(Engineered)* | `state_key` | `string` | Canonical state join key (`telangana`) |
| *(Engineered)* | `discom_code` | `string` | Utility identifier (`TGSPDCL`) |
| `Circle` | `discom_circle` | `string` | DISCOM administrative circle (15 circles) |
| `Division` | `discom_division` | `string` | DISCOM division (24 divisions) |
| `SubDivision` | `discom_subdivision` | `string` | DISCOM subdivision (24 subdivisions) |
| `Section` | `discom_section` | `string` | DISCOM local section / feeder boundary (24 sections) |
| *(Mapped)* | `district` | `string` | Mapped administrative district |
| *(Mapped)* | `district_key` | `string` | Canonical district join key (`slugify(district)`) |
| `Month_Year` | `billing_month` | `string` | Standardized ISO month (`YYYY-MM`) |
| *(Engineered)* | `billing_year` | `int64` | Calendar year (`2023` or `2024`) |
| *(Engineered)* | `financial_year` | `string` | Fiscal year (`FY2023-24`) |
| `Category_Code` | `tariff_category` | `string` | Official TSERC tariff schedule (`LT-IX`) |
| `Total_Services` | `sanctioned_ev_connections` | `int64` | Sanctioned LT-IX connection count |
| `Billed_Services` | `billed_ev_connections` | `int64` | Active billed service connections |
| *(Engineered)* | `contracted_load_kw` | `float64` | Contracted aggregate grid load (kW) |
| `Billed_Load_KW` | `billed_peak_load_kw` | `float64` | Measured monthly peak demand (kW) |
| `Units_Consumed_KWH` | `measured_electricity_consumed_kwh` | `int64` | **Target candidate:** Utility metered consumption (kWh) |
| `Billed_Amount_INR` | `billed_revenue_inr` | `int64` | Billed utility revenue (INR) |
| *(Engineered)* | `average_kwh_per_billed_connection`| `float64` | Consumption intensity per active meter |
| *(Engineered)* | `_provenance` | `string` | Complete audit trail & publishing authority |

---

## 5. Granularity Caveat & Anti-Hallucination Directive

> [!CAUTION]
> **Strict Analytical Constraint:**  
> The TGSPDCL ledger records electricity consumption at the **DISCOM Section / Feeder level**, aggregating between 4 and 16 physical commercial and public EV charging connections within each utility section.  
> **Under NO circumstances may section-level kWh values be divided or disaggregated into synthetic "station-level demand" estimates.**  
> Doing so would create pseudo-telemetry and violate the core gatekeeper principles established in Phase 7. The data must remain at the Section $\times$ Month or District $\times$ FY analytical level.

---
*Audit completed and certified by Antigravity Phase 8 Data Engineering Engine.*
