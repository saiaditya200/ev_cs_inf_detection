# Phase 5 — Government Data Sources Report

## Overview

This report documents all government datasets collected for the Telangana EV Charging Infrastructure Planning project. Each dataset has been sourced from official government portals, press releases, and authoritative open-data platforms.

**Collection Date:** 2026-09-26  
**Total Files:** 10  
**Total Records:** 101  
**Output Directory:** `data/government/`

---

## 1. EV Registration Data

### 1.1 Telangana State EV Registrations
| Attribute | Detail |
|---|---|
| **File** | [`ev_registrations_telangana.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/ev_registrations_telangana.csv) |
| **Records** | 7 (FY2019-20 to FY2025-26 partial) |
| **Source Authority** | VAHAN Public Dashboard (MoRTH) |
| **Portal URL** | https://analytics.parivahan.gov.in/ |
| **Supplementary Sources** | PIB releases, CEEW GFC Dashboard, Times of India, Siasat Daily, Electric Vehicle Talks, EVIndia.Online, Autocar Pro |
| **Geographic Level** | State |
| **Temporal Granularity** | Financial Year (Apr-Mar) |

**Key Observations:**
- EV registrations grew from **1,842** (FY2019-20) to **91,637** (FY2024-25) — a **~50x increase**
- The **100% road tax + registration fee exemption** (Nov 2024) triggered a dramatic surge
- Cumulative registrations reached **~1.58 lakh** by Jun 2026
- Two-wheelers (2W) remain the dominant EV category across all years

> [!IMPORTANT]
> Telangana migrated to VAHAN 4.0 in early 2026. Data for Mar-May 2026 was temporarily unavailable on the national dashboard and was back-filled on 11 May 2026. This may cause minor discrepancies in cross-source comparisons for that period.

### 1.2 All-India EV Registrations (Benchmark)
| Attribute | Detail |
|---|---|
| **File** | [`ev_registrations_india.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/ev_registrations_india.csv) |
| **Records** | 6 (FY2019-20 to FY2024-25) |
| **Source Authority** | VAHAN / SMEV / FADA / PIB |

---

## 2. Population Data

### 2.1 District-Level Population (Census 2011)
| Attribute | Detail |
|---|---|
| **File** | [`population_telangana_districts.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/population_telangana_districts.csv) |
| **Records** | 33 districts |
| **Source Authority** | Census of India 2011 (Primary Census Abstract) |
| **Portal URL** | https://censusindia.gov.in |
| **Supplementary** | Telangana State Statistical Abstract |
| **Geographic Level** | District (post-2016 33-district reorganization) |

**Fields:** Population, area (sq km), density, urban population percentage

> [!NOTE]
> Telangana was carved from Andhra Pradesh in 2014 and reorganized from 10 to 33 districts in 2016. The Census 2011 data has been re-mapped to the current 33-district boundaries using mandal-level disaggregation published in state gazettes.

### 2.2 State-Level Population Projections
| Attribute | Detail |
|---|---|
| **File** | [`population_telangana_projections.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/population_telangana_projections.csv) |
| **Records** | 4 (2011, 2021, 2026, 2031) |
| **Source Authority** | National Commission on Population (NCP) |

---

## 3. EV Charging Infrastructure (Government-Sanctioned)

| Attribute | Detail |
|---|---|
| **File** | [`ev_charging_infrastructure_fame.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/ev_charging_infrastructure_fame.csv) |
| **Records** | 5 (Telangana + India, FAME-II + PM E-DRIVE) |
| **Source Authority** | Ministry of Heavy Industries (MHI) |
| **Portal URL** | https://dash.heavyindustries.gov.in/ |

**Telangana Highlights:**
- **138 stations sanctioned** in cities (Hyderabad, Warangal + others)
- **46 stations sanctioned** on highways (NH 44, NH 65 corridors)
- **~62% installation rate** (85 of 138 city stations installed)

---

## 4. Highway Infrastructure

### 4.1 State-Level Highway Metrics
| Attribute | Detail |
|---|---|
| **File** | [`highway_infrastructure_telangana.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/highway_infrastructure_telangana.csv) |
| **Records** | 5 metrics |
| **Source Authority** | MoRTH / NHAI / PIB |

### 4.2 Major NH Corridors in Telangana
| Attribute | Detail |
|---|---|
| **File** | [`nh_corridors_telangana.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/nh_corridors_telangana.csv) |
| **Records** | 6 corridors |
| **Source Authority** | NHAI / MoRTH |

| NH | Corridor | Length (km) | Lanes |
|---|---|---|---|
| NH 44 | Delhi-Chennai (N-S) | 564 | 4/6 |
| NH 65 | Pune-Machilipatnam (E-W) | 449 | 4/6 |
| NH 163 | Hyderabad-Bhopalpatnam | 312 | 2/4 |
| NH 765 | Hyderabad-Nagarjunasagar | 205 | 2/4 |
| NH 161 | Hyderabad-Nagpur (alt) | 280 | 2/4 |
| NH 167 | Kodad-Khammam-Warangal | 190 | 2 |

**Total NH Network:** 4,983 km (98% growth from 2,511 km in 2014)

---

## 5. Electricity Data

| Attribute | Detail |
|---|---|
| **File** | [`electricity_telangana.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/electricity_telangana.csv) |
| **Records** | 10 (Telangana + India, FY2019-20 to FY2023-24) |
| **Source Authority** | Central Electricity Authority (CEA) |
| **Portal URL** | https://cea.nic.in |

**Telangana per capita consumption:** 1,985 → 2,508 kWh (FY2019-20 to FY2023-24)  
**National average comparison:** Telangana is **~67% above** the national per-capita (1,496 kWh)

---

## 6. EV Policy Milestones

| Attribute | Detail |
|---|---|
| **File** | [`ev_policy_milestones.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/ev_policy_milestones.csv) |
| **Records** | 7 policies (3 state, 4 national) |
| **Source Authority** | Various (MHI, NITI Aayog, MoP, BIS, Telangana IT/Transport Dept) |

Key policies captured:
1. Telangana EV & ESS Policy 2020-2030
2. Road Tax + Registration Fee 100% Exemption (Nov 2024)
3. FAME-II Scheme
4. PM E-DRIVE Scheme
5. Battery Swapping Policy (Draft)
6. BIS EV Charging Standards (IS 17017)
7. MoP Revised Charging Guidelines

---

## 7. Data Dictionary

| Attribute | Detail |
|---|---|
| **File** | [`government_data_dictionary.csv`](file:///c:/Users/Dell/Downloads/EV_Charging/data/government/government_data_dictionary.csv) |
| **Entries** | 18 field-level definitions |

---

## Source Credibility Assessment

| Source | Authority Level | Data Type | Access Method |
|---|---|---|---|
| VAHAN Dashboard | **Official (MoRTH)** | EV registrations | Web dashboard |
| Census of India | **Official (RGI)** | Population | Published tables |
| CEA | **Official (MoP)** | Electricity | Annual reports |
| MHI FAME Dashboard | **Official (MHI)** | Charging infra | Web dashboard |
| NHAI / MoRTH | **Official** | Highways | Reports / PIB |
| PIB | **Official** | Policy | Press releases |
| NCP | **Official** | Pop. projections | Published report |

> [!NOTE]
> All data figures have been cross-referenced against at least two sources where possible. Where exact figures vary between sources (e.g., EV registrations during the Vahan 4.0 migration window), the most conservative and most recently verified figure has been used.

---

## Limitations

1. **No sub-district EV registration data** — VAHAN provides state-level aggregates only via the public dashboard; RTO-level breakdowns require official data requests
2. **Census 2011 is 15+ years old** — District populations may have shifted significantly, especially around Hyderabad
3. **FAME installation counts are point-in-time** — New stations continue to be installed; figures will update
4. **Electricity data excludes EV-specific consumption** — CEA publishes total consumption only; EV-specific load is not disaggregated
5. **Policy data captures key milestones only** — Minor amendments and state notifications may not be exhaustive
