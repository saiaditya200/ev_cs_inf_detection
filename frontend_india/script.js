/* ==========================================================================
   PAN-INDIA EXPLAINABLE AI EV CHARGING PLANNING DASHBOARD - JAVASCRIPT
   100% Dynamic Metrics, Multi-Tab Navigation & Real-Time Inference
   ========================================================================== */

function metricOrDash(value, digits = 4) {
    const n = Number(value);
    return Number.isFinite(n) ? n.toFixed(digits) : "—";
}

document.addEventListener("DOMContentLoaded", () => {
    initNavigation();
    initDataFetching();
    initPredictionForm();
    initPlanningSearch();
});

// ========================= 1. NAVIGATION LOGIC =========================
function initNavigation() {
    const tabs = document.querySelectorAll(".nav-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const tabId = tab.getAttribute("data-tab");
            switchTab(tabId);
        });
    });

    const urlParams = new URLSearchParams(window.location.search);
    const initialTab = urlParams.get("tab") || window.location.hash.replace("#", "");
    if (initialTab) {
        switchTab(initialTab);
    }
}

function switchTab(tabId) {
    document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-pane").forEach(p => p.classList.remove("active"));

    const targetTab = document.querySelector(`.nav-tab[data-tab="${tabId}"]`);
    const targetPane = document.getElementById(`pane-${tabId}`);

    if (targetTab && targetPane) {
        targetTab.classList.add("active");
        targetPane.classList.add("active");
        window.scrollTo({ top: 0, behavior: "smooth" });
    }
}

// ========================= 2. DATA FETCHING & POPULATION =========================
async function initDataFetching() {
    try {
        await Promise.all([
            fetchHealthData(),
            fetchPlanningData(),
            fetchMetricsData(),
            fetchXaiData()
        ]);
    } catch (err) {
        console.error("[ERROR] Failed to fetch initial data:", err);
    }
}

async function fetchHealthData() {
    try {
        const res = await fetch("/api/health");
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById("kpi-stations").textContent = data.station_count ? data.station_count.toLocaleString() : "--";
        document.getElementById("kpi-districts").textContent = data.district_count ?? "--";
        document.getElementById("kpi-district-count-inline").textContent = data.district_count ?? "--";
        document.getElementById("kpi-fast-count").textContent = data.station_class_counts?.dc_fast ? data.station_class_counts.dc_fast.toLocaleString() : "--";
        document.getElementById("kpi-slow-count").textContent = data.station_class_counts?.ac_slow ? data.station_class_counts.ac_slow.toLocaleString() : "--";
        document.getElementById("kpi-unknown-count").textContent = data.station_class_counts?.unknown ?? "0";

        if (data.grid_throughput_gwh !== null && data.grid_throughput_gwh !== undefined) {
            document.getElementById("kpi-energy").innerHTML = `${Number(data.grid_throughput_gwh).toLocaleString()} <small>GWh</small>`;
        } else {
            document.getElementById("kpi-energy").innerHTML = `-- <small>GWh</small>`;
        }

        if (data.cv_pr_auc !== null && data.cv_pr_auc !== undefined) {
            document.getElementById("kpi-prauc").textContent = Number(data.cv_pr_auc).toFixed(4);
        }
        document.getElementById("kpi-tier1").innerHTML = `${data.tier1_districts_count ?? "--"} <small>States</small>`;
        document.getElementById("kpi-features").textContent = data.n_predictors ?? "--";

        if (data.model_name) {
            const runtime = data.sklearn_version ? ` • sklearn ${data.sklearn_version}` : "";
            document.getElementById("model-status-badge").textContent = `Model: ${data.model_name}${runtime}`;
        }
    } catch (e) {
        console.warn("Health API error:", e);
    }
}

async function fetchPlanningData() {
    try {
        const res = await fetch("/api/planning");
        if (!res.ok) return;
        const data = await res.json();

        // 1. Populate State dropdown in prediction form
        const distSelect = document.getElementById("st-district");
        distSelect.innerHTML = "";
        const states = data.district_rankings || [];

        states.forEach(d => {
            const opt = document.createElement("option");
            const stName = d.state || d.district;
            opt.value = stName;
            opt.textContent = `${stName} (Rank #${d.rank}, EIPS: ${parseFloat(d.EIPS).toFixed(1)})`;
            if (stName === "Karnataka") opt.selected = true;
            distSelect.appendChild(opt);
        });

        // 2. Populate 36 States & UTs Rankings Table
        const tbody = document.getElementById("planning-table-body");
        tbody.innerHTML = "";

        states.forEach(r => {
            const tr = document.createElement("tr");
            const rank = parseInt(r.rank);
            const tierBadgeClass = rank <= 10 ? "badge-danger" : rank <= 22 ? "badge-warning" : "badge-success";
            const stName = r.state || r.district;
            tr.innerHTML = `
                <td><b>#${rank}</b></td>
                <td><b>${stName}</b></td>
                <td><span style="font-family:'JetBrains Mono'; font-weight:700;">${parseFloat(r.EIPS).toFixed(1)}</span></td>
                <td><span class="badge ${tierBadgeClass}">${r.priority_tier.split(":")[0]}</span></td>
                <td>${parseFloat(r.district_ev_penetration_rate).toFixed(1)}</td>
                <td><b>${parseFloat(r.infrastructure_gap_index).toFixed(2)}</b></td>
                <td><b>${r.fast_stations.toLocaleString()}</b></td>
                <td>${r.total_stations ? r.total_stations.toLocaleString() : "—"}</td>
                <td>${r.nh_corridors_crossing_count}</td>
                <td style="max-width:320px; font-size:0.78rem; color:#9ca3af; white-space:normal;">${r.eips_explanation}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.warn("Planning API error:", e);
    }
}

async function fetchMetricsData() {
    try {
        const res = await fetch("/api/metrics");
        if (!res.ok) return;
        const data = await res.json();
        const models = data.models || [];

        const tbody = document.getElementById("models-table-body");
        tbody.innerHTML = "";

        models.forEach(m => {
            const tr = document.createElement("tr");
            const isChamp = m.is_champion === true;
            const cv_prauc = metricOrDash(m.cv_pr_auc);
            const outer_prauc = metricOrDash(m.outer_test_pr_auc);
            const outer_roc = metricOrDash(m.outer_test_roc_auc);
            const outer_acc_num = Number(m.outer_test_accuracy);
            const outer_bal_num = Number(m.outer_test_balanced_acc);
            const outer_f1 = metricOrDash(m.outer_test_macro_f1);
            const outer_acc = Number.isFinite(outer_acc_num) ? `${(outer_acc_num * 100).toFixed(1)}%` : "—";
            const outer_bal = Number.isFinite(outer_bal_num) ? `${(outer_bal_num * 100).toFixed(1)}%` : "—";

            tr.innerHTML = `
                <td><b>${m.model_name}</b> ${isChamp ? '<span class="badge badge-primary">Champion</span>' : ''}</td>
                <td><span style="font-family:'JetBrains Mono'; font-weight:700;">${cv_prauc}</span></td>
                <td><span style="font-family:'JetBrains Mono'; font-weight:700;">${outer_prauc}</span></td>
                <td>${outer_roc}</td>
                <td>${outer_acc}</td>
                <td>${outer_bal}</td>
                <td>${outer_f1}</td>
                <td><span class="badge badge-success">Evaluated</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.warn("Metrics API error:", e);
    }
}

async function fetchXaiData() {
    try {
        const res = await fetch("/api/xai");
        if (!res.ok) return;
        const data = await res.json();
        const cfList = data.counterfactual_analysis || [];

        const tbody = document.getElementById("cf-table-body");
        tbody.innerHTML = "";

        if (!cfList.length) {
            tbody.innerHTML = '<tr><td colspan="5" class="text-center">No persisted counterfactual transition records available.</td></tr>';
            return;
        }
        cfList.forEach(r => {
            const original = Number(r.original_prob_fast);
            const counterfactual = Number(r.counterfactual_prob_fast);
            const originalText = Number.isFinite(original) ? `${(original * 100).toFixed(1)}%` : "—";
            const cfText = Number.isFinite(counterfactual) ? `${(counterfactual * 100).toFixed(1)}%` : "—";
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><span style="font-family:'JetBrains Mono'; font-size:0.8rem;">${r.station_id ?? "—"}</span></td>
                <td><b>${r.district || r.state ?? "—"}</b></td>
                <td>${originalText}</td>
                <td><b style="color:var(--success);">${cfText}</b></td>
                <td style="font-size:0.8rem; color:#9ca3af; white-space:normal;">${r.actionable_modifications ?? "—"}</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.warn("XAI API error:", e);
    }
}

// ========================= 3. PREDICTION FORM & INFERENCE =========================
let currentPredictMode = "prospective"; // "prospective" | "existing"
let allStationRecords = [];
let selectedStationRecord = null;

async function initPredictionForm() {
    const form = document.getElementById("prediction-form");
    const modeBtnProspective = document.getElementById("mode-btn-prospective");
    const modeBtnExisting = document.getElementById("mode-btn-existing");
    const formModeTitle = document.getElementById("form-mode-title");
    const modeIndicatorPill = document.getElementById("mode-indicator-pill");
    const prospectiveShortcuts = document.getElementById("prospective-shortcuts");
    const prospectiveNotice = document.getElementById("prospective-note-container");
    const existingContainer = document.getElementById("existing-station-container");
    const stationSelect = document.getElementById("existing-station-select");
    const btnPrefillCorridor = document.getElementById("btn-prefill-corridor");
    const btnDemoKa = document.getElementById("btn-demo-ka");
    const btnDemoMh = document.getElementById("btn-demo-mh");
    const btnDemoTs = document.getElementById("btn-demo-ts");

    // Load sample stations from 52K dataset
    try {
        const res = await fetch("/api/stations");
        if (res.ok) {
            const stData = await res.json();
            allStationRecords = stData.stations || [];
            populateStationDropdown(allStationRecords);
        }
    } catch (err) {
        console.warn("[WARN] Could not fetch station dataset:", err);
    }

    function populateStationDropdown(stations) {
        if (!stationSelect) return;
        stationSelect.innerHTML = `<option value="">-- Choose verified station record (${stations.length} representative records available) --</option>`;
        stations.forEach(s => {
            const opt = document.createElement("option");
            opt.value = s.station_id;
            const name = s.station_name || "Station";
            const st = s.state || s.district;
            opt.textContent = `${s.station_id} — ${name} (${st})`;
            stationSelect.appendChild(opt);
        });
    }

    function setPredictionMode(mode) {
        currentPredictMode = mode;
        if (mode === "prospective") {
            modeBtnProspective.classList.add("active");
            modeBtnExisting.classList.remove("active");
            if (prospectiveShortcuts) prospectiveShortcuts.style.display = "flex";
            if (prospectiveNotice) prospectiveNotice.style.display = "flex";
            if (existingContainer) existingContainer.style.display = "none";
            if (formModeTitle) formModeTitle.textContent = "Prospective Candidate Attributes";
            if (modeIndicatorPill) {
                modeIndicatorPill.textContent = "Input Mode: Prospective Site";
                modeIndicatorPill.style.background = "rgba(59, 130, 246, 0.15)";
                modeIndicatorPill.style.color = "#93c5fd";
                modeIndicatorPill.style.borderColor = "rgba(59, 130, 246, 0.3)";
            }
            selectedStationRecord = null;
        } else {
            modeBtnExisting.classList.add("active");
            modeBtnProspective.classList.remove("active");
            if (prospectiveShortcuts) prospectiveShortcuts.style.display = "none";
            if (prospectiveNotice) prospectiveNotice.style.display = "none";
            if (existingContainer) existingContainer.style.display = "block";
            if (formModeTitle) formModeTitle.textContent = "Existing Station Site Attributes";
            if (modeIndicatorPill) {
                modeIndicatorPill.textContent = "Input Mode: Existing Station Record";
                modeIndicatorPill.style.background = "rgba(16, 185, 129, 0.15)";
                modeIndicatorPill.style.color = "#6ee7b7";
                modeIndicatorPill.style.borderColor = "rgba(16, 185, 129, 0.3)";
            }
            if (!selectedStationRecord && allStationRecords.length > 0) {
                loadStationById(allStationRecords[0].station_id);
            }
        }
    }

    function loadStationById(stationId) {
        const found = allStationRecords.find(s => s.station_id === stationId);
        if (!found) return;
        selectedStationRecord = found;
        if (stationSelect) stationSelect.value = stationId;

        document.getElementById("st-id").value = found.station_id || "";
        document.getElementById("st-station-name").value = found.station_name || "";
        document.getElementById("st-district").value = found.state || found.district || "Karnataka";
        document.getElementById("st-lat").value = Number(found.latitude).toFixed(6);
        document.getElementById("st-lon").value = Number(found.longitude).toFixed(6);
        if (found.road_type) {
            document.getElementById("st-road").value = found.road_type.toLowerCase();
        }
        document.getElementById("st-hwy-dist").value = Number(found.distance_to_highway_km || 1.2).toFixed(2);
        document.getElementById("st-city-dist").value = Number(found.distance_to_city_center_km || 4.5).toFixed(2);
        document.getElementById("st-poi-count").value = Math.round(Number(found.nearby_poi_count) || 12);
        document.getElementById("st-poi-div").value = Number(found.poi_diversity_index || 0.70).toFixed(2);
    }

    function loadProspectiveCorridor() {
        selectedStationRecord = null;
        document.getElementById("st-id").value = "IN_PROSPECTIVE_CORRIDOR_01";
        document.getElementById("st-station-name").value = "Illustrative National Highway Commercial Hub";
        document.getElementById("st-district").value = "Karnataka";
        document.getElementById("st-lat").value = 12.9716;
        document.getElementById("st-lon").value = 77.5946;
        document.getElementById("st-road").value = "primary";
        document.getElementById("st-hwy-dist").value = 0.75;
        document.getElementById("st-city-dist").value = 4.8;
        document.getElementById("st-poi-count").value = 16;
        document.getElementById("st-poi-div").value = 0.82;
    }

    if (modeBtnProspective) {
        modeBtnProspective.addEventListener("click", () => {
            setPredictionMode("prospective");
            loadProspectiveCorridor();
        });
    }

    if (modeBtnExisting) {
        modeBtnExisting.addEventListener("click", () => {
            setPredictionMode("existing");
        });
    }

    if (stationSelect) {
        stationSelect.addEventListener("change", (e) => {
            if (e.target.value) {
                loadStationById(e.target.value);
            }
        });
    }

    if (btnPrefillCorridor) {
        btnPrefillCorridor.addEventListener("click", loadProspectiveCorridor);
    }

    if (btnDemoKa) {
        btnDemoKa.addEventListener("click", () => {
            setPredictionMode("existing");
            const st = allStationRecords.find(s => (s.state === "Karnataka") || s.station_id.includes("KA"));
            if (st) loadStationById(st.station_id);
        });
    }

    if (btnDemoMh) {
        btnDemoMh.addEventListener("click", () => {
            setPredictionMode("existing");
            const st = allStationRecords.find(s => (s.state === "Maharashtra") || s.station_id.includes("MH"));
            if (st) loadStationById(st.station_id);
        });
    }

    if (btnDemoTs) {
        btnDemoTs.addEventListener("click", () => {
            setPredictionMode("existing");
            const st = allStationRecords.find(s => (s.state === "Telangana") || s.station_id.includes("TS") || s.station_id.includes("MED"));
            if (st) loadStationById(st.station_id);
        });
    }

    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const btn = document.getElementById("btn-submit-predict");
        const spinner = document.getElementById("predict-spinner");
        btn.disabled = true;
        spinner.style.display = "inline-block";

        const payload = {
            station_id: document.getElementById("st-id").value,
            station_name: document.getElementById("st-station-name").value,
            district: document.getElementById("st-district").value,
            latitude: parseFloat(document.getElementById("st-lat").value),
            longitude: parseFloat(document.getElementById("st-lon").value),
            road_type: document.getElementById("st-road").value,
            distance_to_highway_km: parseFloat(document.getElementById("st-hwy-dist").value),
            distance_to_city_center_km: parseFloat(document.getElementById("st-city-dist").value),
            nearby_poi_count: parseFloat(document.getElementById("st-poi-count").value),
            poi_diversity_index: parseFloat(document.getElementById("st-poi-div").value)
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.message || "Prediction error");
            }

            const data = await res.json();
            renderPredictionResults(data, selectedStationRecord, currentPredictMode);
        } catch (err) {
            console.error("[ERROR] Inference Failure:", err);
            const empty = document.getElementById("res-empty-state");
            if (empty) {
                empty.innerHTML = `<div class="empty-state-icon">⚠️</div><h4>Inference Failed</h4><p>${String(err.message).replace(/</g, "&lt;")}</p>`;
            }
            alert(`Inference Failure: ${err.message}`);
        } finally {
            btn.disabled = false;
            spinner.style.display = "none";
        }
    });
}

function renderPredictionResults(data, observedMeta, mode) {
    const emptyCard = document.getElementById("res-empty-state-card");
    if (emptyCard) emptyCard.style.display = "none";

    const resContent = document.getElementById("res-content");
    if (resContent) resContent.style.display = "flex";

    const pred = data.prediction;
    const pFast = Number(pred.probability_dc_fast);
    const pSlow = Number(pred.probability_ac_slow);
    const threshold = Number(pred.threshold_applied) || 0.2488;

    const isFast = pFast >= threshold;
    const conf = Number(pred.predicted_class_probability) || (isFast ? pFast : pSlow);

    // ==================== CARD 1: MODEL INFERENCE & OBSERVED REFERENCE ====================
    document.getElementById("res-p-fast").textContent = `${(pFast * 100).toFixed(1)}%`;
    document.getElementById("res-p-slow").textContent = `${(pSlow * 100).toFixed(1)}%`;
    document.getElementById("res-pred-confidence").textContent = `${(conf * 100).toFixed(1)}%`;
    document.getElementById("res-thresh-val").textContent = `${(threshold * 100).toFixed(1)}%`;

    const meterFill = document.getElementById("res-meter-fill");
    if (meterFill) {
        meterFill.style.width = `${Math.min(100, Math.max(0, pFast * 100)).toFixed(1)}%`;
        meterFill.style.background = isFast
            ? "linear-gradient(90deg, #06b6d4, #10b981)"
            : "linear-gradient(90deg, #ef4444, #f59e0b)";
    }

    const banner = document.getElementById("pred-class-banner");
    if (banner) {
        banner.className = `pred-class-banner ${isFast ? 'pred-fast' : 'pred-slow'}`;
    }

    const classNameEl = document.getElementById("res-class-name");
    if (classNameEl) {
        classNameEl.textContent = isFast
            ? "DC Fast-Charging Infrastructure (>= 50 kW)"
            : "AC Slow / Destination Charging Infrastructure (< 50 kW)";
    }

    const statusBadge = document.getElementById("res-status-badge");
    if (statusBadge) {
        statusBadge.textContent = isFast ? "Classified: DC Fast" : "Classified: AC Slow";
        statusBadge.className = `badge badge-sm ${isFast ? 'badge-success' : 'badge-warning'}`;
    }

    // Baseline Transfer Check Comparison
    const tgCompBadge = document.getElementById("res-tg-comp-badge");
    if (tgCompBadge && data.telangana_baseline_transfer) {
        const tgP = data.telangana_baseline_transfer.probability_dc_fast;
        const tgIsFast = data.telangana_baseline_transfer.predicted_class_code === 1;
        tgCompBadge.textContent = `Telangana Baseline P(Fast): ${(tgP * 100).toFixed(1)}% (${tgIsFast ? 'Fast' : 'Slow'})`;
        tgCompBadge.className = `badge badge-sm ${tgIsFast === isFast ? 'badge-info' : 'badge-warning'}`;
    }

    // Observed Dataset Reference
    const observedBadge = document.getElementById("res-observed-badge");
    const observedNote = document.getElementById("res-observed-note");

    if (mode === "existing" && observedMeta) {
        const flag = observedMeta.fast_charging_flag;
        if (flag === 1 || flag === 1.0 || flag === "1") {
            observedBadge.textContent = "DC Fast (>= 50 kW)";
            observedBadge.className = "observed-badge badge-success";
            observedNote.innerHTML = `Observed ground-truth label from verified station record <strong>${observedMeta.station_id}</strong> in master 52K dataset. Not used as a model input.`;
        } else if (flag === 0 || flag === 0.0 || flag === "0") {
            observedBadge.textContent = "AC / Non-Fast (observed)";
            observedBadge.className = "observed-badge badge-amber";
            observedNote.innerHTML = `Observed ground-truth label from verified station record <strong>${observedMeta.station_id}</strong> in master 52K dataset. Not used as a model input.`;
        } else {
            observedBadge.textContent = "Unlabeled / Unknown";
            observedBadge.className = "observed-badge badge-neutral";
            observedNote.innerHTML = `Observed classification: <strong>Unlabeled / Unknown</strong> in master dataset.`;
        }
    } else {
        observedBadge.textContent = "N/A (Prospective Site)";
        observedBadge.className = "observed-badge badge-neutral";
        observedNote.textContent = "User-defined planning scenario — not an observed station record.";
    }

    // ==================== CARD 2: WHY? — LOCAL TREESHAP ====================
    const posDiv = document.getElementById("res-positive-drivers");
    if (posDiv) {
        posDiv.innerHTML = "";
        const posDrivers = data.explainable_ai?.top_positive_drivers || [];
        if (posDrivers.length === 0) {
            posDiv.innerHTML = '<span style="font-size:0.75rem; color:#9ca3af;">None identified</span>';
        } else {
            posDrivers.forEach(d => {
                const item = document.createElement("div");
                item.className = "driver-bar-item";
                item.innerHTML = `
                    <span class="driver-name">${d.feature} (= ${d.value})</span>
                    <span class="driver-shap text-success">+${d.shap_impact.toFixed(4)}</span>
                `;
                posDiv.appendChild(item);
            });
        }
    }

    const negDiv = document.getElementById("res-negative-drivers");
    if (negDiv) {
        negDiv.innerHTML = "";
        const negDrivers = data.explainable_ai?.top_negative_drivers || [];
        if (negDrivers.length === 0) {
            negDiv.innerHTML = '<span style="font-size:0.75rem; color:#9ca3af;">None identified</span>';
        } else {
            negDrivers.forEach(d => {
                const item = document.createElement("div");
                item.className = "driver-bar-item";
                item.innerHTML = `
                    <span class="driver-name">${d.feature} (= ${d.value})</span>
                    <span class="driver-shap text-danger">${d.shap_impact.toFixed(4)}</span>
                `;
                negDiv.appendChild(item);
            });
        }
    }

    // ==================== CARD 3: WHAT-IF PLANNING ====================
    const cfContainer = document.getElementById("res-cf-simulation-content");
    if (cfContainer) {
        const cf = data.actionable_counterfactual || {};
        if (cf.status === "already_above_threshold") {
            cfContainer.innerHTML = `
                <div class="cf-card-box">
                    <div class="cf-meta-row">
                        <span>Current P(DC Fast): <b class="text-cyan">${(pFast * 100).toFixed(1)}%</b></span>
                        <span class="cf-badge-crossed">Above Threshold (&ge; ${(threshold * 100).toFixed(1)}%)</span>
                    </div>
                    <p style="font-size:0.76rem; color:#9ca3af; margin:4px 0 0 0; line-height:1.4;">
                        Current physical site attributes already satisfy the national decision threshold for DC Fast infrastructure classification.
                    </p>
                </div>
            `;
        } else if (cf.best_scenario) {
            const best = cf.best_scenario;
            const lever = best.changed_lever;
            let origVal = "Current";
            if (data.feature_vector && data.feature_vector[lever] !== undefined) {
                origVal = data.feature_vector[lever];
            } else if (lever === "distance_to_highway_km") {
                origVal = document.getElementById("st-hwy-dist").value;
            }

            const newVal = typeof best.new_value === 'object' ? JSON.stringify(best.new_value) : best.new_value;
            const resProb = (best.probability_dc_fast * 100).toFixed(1);
            const crossed = best.crosses_threshold;

            let html = `
                <div class="cf-card-box">
                    <div class="cf-meta-row">
                        <span class="cf-lever-name">Simulated Lever: <b>${lever}</b></span>
                        <span class="cf-result-prob ${crossed ? 'text-success' : 'text-amber'}">${resProb}% P(DC Fast)</span>
                    </div>
                    <div class="cf-vals-row">
                        <span>Original: <b>${origVal}</b> &rarr; Modified: <b>${newVal}</b></span>
                        <span class="${crossed ? 'cf-badge-crossed' : 'cf-badge-not-crossed'}">${crossed ? 'Crosses Decision Threshold' : 'Below Threshold'}</span>
                    </div>
                </div>
            `;

            if (cf.scenarios && cf.scenarios.length > 1) {
                html += '<div style="margin-top:8px;"><span style="font-size:0.72rem; color:#9ca3af; text-transform:uppercase; font-weight:600;">Alternative Model Scenarios:</span></div>';
                cf.scenarios.slice(1, 3).forEach(sc => {
                    const scNew = typeof sc.new_value === 'object' ? JSON.stringify(sc.new_value) : sc.new_value;
                    const scP = (sc.probability_dc_fast * 100).toFixed(1);
                    html += `
                        <div class="cf-card-box" style="padding:6px 10px; margin-top:4px;">
                            <div class="cf-meta-row" style="margin-bottom:0;">
                                <span style="font-size:0.75rem; font-family:'JetBrains Mono';">${sc.changed_lever} &rarr; ${scNew}</span>
                                <span style="font-size:0.75rem; font-weight:700;" class="${sc.crosses_threshold ? 'text-success' : 'text-amber'}">${scP}%</span>
                            </div>
                        </div>
                    `;
                });
            }

            if (cf.key_recommendations && cf.key_recommendations.length) {
                html += '<ul class="cf-recs-list">';
                cf.key_recommendations.forEach(rec => {
                    html += `<li>${rec}</li>`;
                });
                html += '</ul>';
            }

            cfContainer.innerHTML = html;
        } else {
            cfContainer.innerHTML = '<div style="font-size:0.78rem; color:#9ca3af;">No counterfactual scenarios available for this configuration.</div>';
        }
    }

    // ==================== CARD 4: REGIONAL STATE PLANNING CONTEXT ====================
    const rc = data.regional_planning_context;
    if (rc && (rc.district || rc.state)) {
        document.getElementById("res-dist-name").textContent = rc.state || rc.district;
        document.getElementById("res-dist-eips").textContent = `${Number(rc.eips_score).toFixed(1)} / 100`;
        document.getElementById("res-dist-rank").textContent = `#${rc.eips_rank} of 36`;
        document.getElementById("res-dist-tier").textContent = rc.priority_tier ? rc.priority_tier.split(":")[0] : "—";
        document.getElementById("res-dist-gap").textContent = Number(rc.infrastructure_gap_index).toFixed(2);
        document.getElementById("res-dist-fast-stns").textContent = Number(rc.existing_fast_stations).toLocaleString() ?? "0";
    }
}

// ========================= 4. PLANNING TABLE FILTER =========================
function initPlanningSearch() {
    const searchInput = document.getElementById("planning-search");
    if (!searchInput) return;

    searchInput.addEventListener("input", () => {
        const q = searchInput.value.toLowerCase();
        const rows = document.querySelectorAll("#planning-table-body tr");
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(q) ? "" : "none";
        });
    });
}

// ========================= 5. IMAGE MODAL =========================
function openModal(src) {
    const modal = document.getElementById("image-modal");
    const modalImg = document.getElementById("modal-img");
    modal.style.display = "block";
    modalImg.src = src;
}

function closeModal() {
    document.getElementById("image-modal").style.display = "none";
}
