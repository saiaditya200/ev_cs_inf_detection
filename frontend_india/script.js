document.addEventListener("DOMContentLoaded", () => {
  let map = null;
  let markersLayer = null;

  // Initialize Map centered on India
  initMap();

  // Load National Overview & Metrics
  loadOverview();

  // Load State List
  loadStates();

  // Setup Event Listeners
  document.getElementById("state-select").addEventListener("change", (e) => {
    const stateName = e.target.value;
    if (stateName) {
      loadStateDetail(stateName);
    }
  });

  document.getElementById("btn-predict").addEventListener("click", runDualPrediction);

  function initMap() {
    map = L.map("map").setView([20.5937, 78.9629], 5);
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 18
    }).addTo(map);

    markersLayer = L.layerGroup().addTo(map);
  }

  async function loadOverview() {
    try {
      const res = await fetch("/api/overview");
      const data = await res.json();
      if (data.status === "success") {
        const sum = data.national_summary;
        document.getElementById("kpi-total").innerText = sum.total_stations.toLocaleString();
        document.getElementById("kpi-fast").innerText = sum.fast_charging_stations.toLocaleString();
        document.getElementById("kpi-slow").innerText = sum.slow_charging_stations.toLocaleString();

        const comp = data.model_comparison;
        if (comp.model_1_telangana_baseline_transfer) {
          const m1 = comp.model_1_telangana_baseline_transfer;
          document.getElementById("tg-acc").innerText = (m1.accuracy * 100).toFixed(2) + "%";
          document.getElementById("tg-auc").innerText = m1.roc_auc.toFixed(4);
          document.getElementById("tg-pr-auc").innerText = m1.pr_auc.toFixed(4);
        }
        if (comp.model_2_pan_india_national_model) {
          const m2 = comp.model_2_pan_india_national_model;
          document.getElementById("nat-pr-auc").innerText = m2.pr_auc.toFixed(4);
          document.getElementById("nat-f1").innerText = m2.f1_score.toFixed(4);
          document.getElementById("nat-thresh").innerText = m2.decision_threshold.toFixed(4);
        }
      }
    } catch (err) {
      console.error("Failed to load overview:", err);
    }
  }

  async function loadStates() {
    try {
      const res = await fetch("/api/states");
      const data = await res.json();
      if (data.status === "success") {
        const select = document.getElementById("state-select");
        select.innerHTML = '<option value="">-- Choose State or UT (36 Available) --</option>';
        data.states.forEach((st) => {
          const opt = document.createElement("option");
          opt.value = st.state_or_ut;
          opt.innerText = `${st.national_rank}. ${st.state_or_ut} (${st.july2026_scaled_total.toLocaleString()} stations)`;
          select.appendChild(opt);
        });

        // Auto-select Karnataka (Rank 1) or Telangana
        if (data.states.length > 0) {
          select.value = "Karnataka";
          loadStateDetail("Karnataka");
        }
      }
    } catch (err) {
      console.error("Failed to load states:", err);
    }
  }

  async function loadStateDetail(stateName) {
    try {
      const res = await fetch(`/api/state/${encodeURIComponent(stateName)}`);
      const data = await res.json();
      if (data.status === "success") {
        document.getElementById("stat-state-title").innerText = `${data.state_name} EV Infrastructure`;
        document.getElementById("stat-total").innerText = data.total_stations.toLocaleString();
        document.getElementById("stat-fast").innerText = data.fast_stations.toLocaleString();
        document.getElementById("stat-slow").innerText = data.slow_stations.toLocaleString();
        document.getElementById("stat-pct").innerText = `${data.fast_pct}%`;

        if (data.performance && data.performance.state_nodal_agency) {
          document.getElementById("stat-agency").innerText = data.performance.state_nodal_agency;
        } else {
          document.getElementById("stat-agency").innerText = "State Designated Nodal Agency";
        }

        // Render Stations in Table
        const tbody = document.getElementById("stations-tbody");
        tbody.innerHTML = "";
        markersLayer.clearLayers();

        const latLngs = [];

        data.stations_sample.forEach((st) => {
          const tr = document.createElement("tr");
          const isFast = st.fast_charging_flag === 1;
          tr.innerHTML = `
            <td><code>${st.station_id}</code></td>
            <td>${st.operator}</td>
            <td><span class="badge ${isFast ? 'badge-success' : 'badge-warning'}">${isFast ? 'DC Fast' : 'AC Slow'}</span></td>
            <td>${st.rated_power_kw} kW</td>
            <td><small>Operational</small></td>
          `;
          tbody.appendChild(tr);

          // Map Marker
          const lat = parseFloat(st.latitude);
          const lon = parseFloat(st.longitude);
          if (!isNaN(lat) && !isNaN(lon)) {
            latLngs.push([lat, lon]);
            const color = isFast ? "#10b981" : "#a855f7";
            const circle = L.circleMarker([lat, lon], {
              radius: 5,
              fillColor: color,
              color: "#fff",
              weight: 1,
              opacity: 1,
              fillOpacity: 0.8
            });

            circle.bindPopup(`
              <div style="font-family:Inter,sans-serif;font-size:12px;">
                <strong>${st.station_name}</strong><br/>
                <b>Operator:</b> ${st.operator}<br/>
                <b>Power:</b> ${st.rated_power_kw} kW (${isFast ? 'DC Fast' : 'AC Slow'})<br/>
                <b>Distance to NH:</b> ${st.distance_to_highway_km} km<br/>
                <b>Road Type:</b> ${st.road_type}
              </div>
            `);
            markersLayer.addLayer(circle);
          }
        });

        if (latLngs.length > 0) {
          const bounds = L.latLngBounds(latLngs);
          map.fitBounds(bounds, { padding: [30, 30] });
        }
      }
    } catch (err) {
      console.error("Failed to load state detail:", err);
    }
  }

  async function runDualPrediction() {
    const lat = parseFloat(document.getElementById("sim-lat").value);
    const lon = parseFloat(document.getElementById("sim-lon").value);
    const hdist = parseFloat(document.getElementById("sim-hdist").value);
    const road = document.getElementById("sim-road").value;
    const poi = parseInt(document.getElementById("sim-poi").value);

    const btn = document.getElementById("btn-predict");
    btn.disabled = true;
    btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Predicting...';

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: jsonPayload({
          latitude: lat,
          longitude: lon,
          distance_to_highway_km: hdist,
          road_type: road,
          nearby_poi_count: poi
        })
      });

      const data = await res.json();
      if (data.status === "success") {
        const tg = data.telangana_baseline_result;
        const nat = data.pan_india_national_result;

        // Update Telangana Box
        document.getElementById("res-pred-tg").innerText = tg.prediction_class;
        document.getElementById("res-prob-tg").innerText = `${tg.fast_charging_probability_pct} (Threshold: ${(tg.decision_threshold * 100).toFixed(0)}%)`;
        document.getElementById("res-bar-tg").style.width = tg.fast_charging_probability_pct;

        // Update National Box
        document.getElementById("res-pred-nat").innerText = nat.prediction_class;
        document.getElementById("res-prob-nat").innerText = `${nat.fast_charging_probability_pct} (Threshold: ${(nat.decision_threshold * 100).toFixed(1)}%)`;
        document.getElementById("res-bar-nat").style.width = nat.fast_charging_probability_pct;
      }
    } catch (err) {
      console.error("Dual prediction failed:", err);
    } finally {
      btn.disabled = false;
      btn.innerHTML = '<i class="fa-solid fa-bolt"></i> Run Dual-Model Inference';
    }
  }

  function jsonPayload(obj) {
    return JSON.stringify(obj);
  }
});
