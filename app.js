/*
  RISKGRAPH — Complete Customer + Order + Entity Case Intelligence Dashboard
  Step 4.4D Implementation
  Connects to FastAPI backend at http://127.0.0.1:8000
*/

const API_BASE_URL = "http://127.0.0.1:8000";

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('assessment-form');
  const inputOrderId = document.getElementById('input-order-id');
  const btnSubmit = document.getElementById('btn-submit-assess');
  const btnText = document.getElementById('btn-text');
  const btnSpinner = document.getElementById('btn-spinner');

  // State Containers
  const placeholderState = document.getElementById('placeholder-state');
  const errorCard = document.getElementById('error-card');
  const errorMessage = document.getElementById('error-message');
  const assessmentResult = document.getElementById('assessment-result');
  const heroCard = document.getElementById('summary-hero-card');

  // Demo Case Buttons
  const btnDemoLow = document.getElementById('btn-demo-low');
  const btnDemoMed = document.getElementById('btn-demo-med');
  const btnDemoHigh = document.getElementById('btn-demo-high');

  // Initial Health Ping
  checkBackendHealth();

  function checkBackendHealth() {
    fetch(`${API_BASE_URL}/health`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data) {
          console.log(`[RISKGRAPH] Backend Connected: ${data.model} (${data.model_version})`);
        }
      })
      .catch(err => {
        console.warn('[RISKGRAPH] Backend health check notice:', err.message);
      });
  }

  // Demo Case Handlers
  if (btnDemoLow) {
    btnDemoLow.addEventListener('click', () => {
      inputOrderId.value = 'ORD100028';
      assessRisk('ORD100028');
    });
  }

  if (btnDemoMed) {
    btnDemoMed.addEventListener('click', () => {
      inputOrderId.value = 'ORD100007';
      assessRisk('ORD100007');
    });
  }

  if (btnDemoHigh) {
    btnDemoHigh.addEventListener('click', () => {
      inputOrderId.value = 'ORD100006';
      assessRisk('ORD100006');
    });
  }

  // Form Submit Handler
  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const orderId = inputOrderId.value.trim();
    if (!orderId) {
      showError("Please enter a valid Order ID.");
      return;
    }
    assessRisk(orderId);
  });

  // Core Assessment API Call
  function assessRisk(orderId) {
    if (!orderId) {
      showError("Please enter a valid Order ID.");
      return;
    }

    setLoadingState(true);

    fetch(`${API_BASE_URL}/assess`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ order_id: orderId })
    })
    .then(async (response) => {
      if (response.ok) {
        return response.json();
      }

      if (response.status === 404) {
        throw new Error("Order not found in database. Please verify the Order ID (e.g. ORD100007, ORD100028, ORD100006).");
      } else if (response.status === 422) {
        throw new Error("Please enter a valid Order ID.");
      } else if (response.status >= 500) {
        throw new Error("Risk assessment could not be completed due to a backend error. Please try again.");
      } else {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || "Unable to complete risk assessment.");
      }
    })
    .then((data) => {
      renderCompleteDashboard(data);
    })
    .catch((err) => {
      console.error('[ASSESSMENT ERROR]', err);
      if (err instanceof TypeError && err.message.toLowerCase().includes('failed to fetch')) {
        showError("Unable to connect to RiskGraph backend. Please ensure the backend is running at http://127.0.0.1:8000.");
      } else {
        showError(err.message || "Unable to connect to RiskGraph backend.");
      }
    })
    .finally(() => {
      setLoadingState(false);
    });
  }

  function setLoadingState(isLoading) {
    if (isLoading) {
      btnSubmit.disabled = true;
      btnSpinner.classList.remove('hidden');
      btnText.textContent = "Analyzing...";
    } else {
      btnSubmit.disabled = false;
      btnSpinner.classList.add('hidden');
      btnText.textContent = "Assess Risk";
    }
  }

  function showError(msg) {
    placeholderState.classList.add('hidden');
    assessmentResult.classList.add('hidden');
    errorMessage.textContent = msg;
    errorCard.classList.remove('hidden');
  }

  // Format Helpers
  function formatINR(num) {
    if (num === null || num === undefined || isNaN(num)) return "₹0.00";
    return "₹" + Number(num).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  }

  function formatPct(num) {
    if (num === null || num === undefined || isNaN(num)) return "0.0%";
    return (Number(num) * 100).toFixed(1) + "%";
  }

  // -------------------------------------------------------------
  // MASTER DASHBOARD RENDERER
  // -------------------------------------------------------------
  function renderCompleteDashboard(data) {
    placeholderState.classList.add('hidden');
    errorCard.classList.add('hidden');
    assessmentResult.classList.remove('hidden');

    const score = data.risk_score || 0.0;
    const tier = data.risk_level || 'LOW';
    const profile = data.customer_profile || {};
    const order = data.current_order || {};
    const bh = data.behavioural_history || {};
    const network = data.entity_network || {};
    const timeline = data.timeline || [];
    const standsOut = data.what_stands_out || [];
    const signals = data.structured_signals || [];
    const ds = data.decision_support || {};

    // 1. CASE SUMMARY HERO
    renderCaseSummary(data, score, tier, profile, order, network);

    // 2. CURRENT REFUND REQUEST
    renderCurrentOrder(order, data.request_id);

    // 3. CUSTOMER PROFILE
    renderCustomerProfile(profile);

    // 4. BEHAVIOURAL ACTIVITY
    renderBehaviouralActivity(bh);

    // 5. CUSTOMER TIMELINE
    renderTimeline(timeline);

    // 6. ENTITY NETWORK & TOPOLOGY GRAPH
    renderEntityNetwork(network, profile.customer_id);

    // 7. WHAT STANDS OUT
    renderWhatStandsOut(standsOut);

    // 8. WHY THIS ASSESSMENT (STRUCTURED SIGNALS)
    renderStructuredSignals(signals, data.top_signals);

    // 9. REFUND DECISION SUPPORT
    renderDecisionSupport(ds, tier);

    // Scroll smoothly into view
    assessmentResult.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // 1. Summary Hero Renderer
  function renderCaseSummary(data, score, tier, profile, order, network) {
    heroCard.className = 'case-summary-card';
    
    document.getElementById('sum-order-id').textContent = data.request_id || 'ORD-UNKNOWN';
    document.getElementById('sum-customer-id').textContent = `Customer ${profile.customer_id || 'CUST-UNKNOWN'}`;
    document.getElementById('sum-customer-segment').textContent = profile.customer_segment || 'Mass Market';

    const scorePct = (score * 100).toFixed(2) + "%";
    document.getElementById('sum-score-pct').textContent = scorePct;
    document.getElementById('sum-score-pval').textContent = `(p = ${score.toFixed(4)})`;

    const tierBadge = document.getElementById('sum-tier-badge');
    const actionPill = document.getElementById('sum-action-pill');
    const safetyGuard = document.getElementById('safety-guard-notice');
    const meterPinTag = document.getElementById('meter-pin-tag');

    tierBadge.className = 'tier-pill';
    actionPill.className = 'action-highlight-pill';

    if (tier === 'LOW') {
      heroCard.classList.add('tier-low-hero');
      tierBadge.classList.add('badge-low');
      tierBadge.textContent = "LOW RISK";
      actionPill.textContent = "NORMAL REFUND PROCESSING";
      actionPill.classList.add('pill-action-low');
      safetyGuard.classList.add('hidden');
    } else if (tier === 'MEDIUM') {
      heroCard.classList.add('tier-med-hero');
      tierBadge.classList.add('badge-med');
      tierBadge.textContent = "MEDIUM RISK";
      actionPill.textContent = "ADDITIONAL VERIFICATION";
      actionPill.classList.add('pill-action-med');
      safetyGuard.classList.add('hidden');
    } else {
      heroCard.classList.add('tier-high-hero');
      tierBadge.classList.add('badge-high');
      tierBadge.textContent = "HIGH RISK";
      actionPill.textContent = "HUMAN SPECIALIST REVIEW";
      actionPill.classList.add('pill-action-high');
      safetyGuard.classList.remove('hidden');
    }

    // Summary Metric Sub-values
    document.getElementById('sum-order-val').textContent = formatINR(order.amount || 0);
    document.getElementById('sum-order-cat-channel').textContent = `${order.category || 'General'} · ${order.channel || 'Direct'}`;
    
    document.getElementById('sum-tenure-val').textContent = `${(profile.customer_tenure_days || 0).toLocaleString()} days`;
    document.getElementById('sum-tenure-sub').textContent = `${profile.prior_order_count || 0} historical orders`;

    const linkedAccCount = network.total_linked_external_accounts || 0;
    document.getElementById('sum-linked-accounts-val').textContent = `${linkedAccCount} ${linkedAccCount === 1 ? 'account' : 'accounts'}`;

    // Risk Meter Pin positioning
    const pinPos = Math.min(97, Math.max(3, score * 100));
    document.getElementById('risk-score-pin').style.left = `${pinPos}%`;
    meterPinTag.textContent = scorePct;
  }

  // 2. Current Order Card Renderer
  function renderCurrentOrder(order, reqId) {
    document.getElementById('ord-id').textContent = order.order_id || reqId;
    document.getElementById('ord-amount').textContent = formatINR(order.amount || 0);
    document.getElementById('ord-qty').textContent = `${order.quantity || 1} item${(order.quantity || 1) > 1 ? 's' : ''}`;
    document.getElementById('ord-category').textContent = order.category || 'General';
    document.getElementById('ord-payment').textContent = order.payment_method || 'Credit Card';
    document.getElementById('ord-channel').textContent = order.channel || 'Website';
    document.getElementById('ord-date').textContent = order.order_date || '—';
    document.getElementById('ord-ship-date').textContent = order.shipping_date || '—';
    document.getElementById('ord-deliv-date').textContent = order.delivery_date || order.shipping_date || '—';
    document.getElementById('ord-ret-date').textContent = `${order.return_request_date || '—'} (Cutoff T)`;
    document.getElementById('ord-reason').textContent = order.return_reason || 'Dispute filed';
  }

  // 3. Customer Profile Renderer
  function renderCustomerProfile(profile) {
    document.getElementById('cp-customer-id').textContent = profile.customer_id || '—';
    document.getElementById('cp-segment').textContent = profile.customer_segment || 'Mass Market';
    document.getElementById('cp-tenure').textContent = `${(profile.customer_tenure_days || 0).toLocaleString()} days`;
    document.getElementById('cp-prior-orders').textContent = `${profile.prior_order_count || 0} orders`;
    document.getElementById('cp-prior-returns').textContent = `${profile.prior_return_count || 0} returns`;
    document.getElementById('cp-prior-refunds').textContent = `${profile.prior_refund_count || 0} refunds`;
    document.getElementById('cp-prior-spend').textContent = formatINR(profile.prior_spend || 0);
    document.getElementById('cp-refund-spend').textContent = formatINR(profile.prior_refund_amount || 0);
    document.getElementById('cp-ret-rate').textContent = formatPct(profile.prior_return_rate || 0);
    document.getElementById('cp-ref-rate').textContent = formatPct(profile.prior_refund_rate || 0);
    document.getElementById('cp-avg-order').textContent = formatINR(profile.average_previous_order_value || 0);
  }

  // 4. Behavioural Activity Renderer
  function renderBehaviouralActivity(bh) {
    document.getElementById('v7-orders').textContent = bh.orders_last_7_days ?? 0;
    document.getElementById('v7-returns').textContent = bh.returns_last_7_days ?? 0;
    document.getElementById('v7-refunds').textContent = bh.refunds_last_7_days ?? 0;

    document.getElementById('v14-orders').textContent = bh.orders_last_14_days ?? 0;
    document.getElementById('v14-returns').textContent = bh.returns_last_14_days ?? 0;
    document.getElementById('v14-refunds').textContent = bh.refunds_last_14_days ?? 0;

    document.getElementById('v30-orders').textContent = bh.orders_last_30_days ?? 0;
    document.getElementById('v30-returns').textContent = bh.returns_last_30_days ?? 0;
    document.getElementById('v30-refunds').textContent = bh.refunds_last_30_days ?? 0;

    const daysPrevOrd = bh.days_since_previous_order;
    document.getElementById('bh-days-prev-ord').textContent = (daysPrevOrd >= 900 || daysPrevOrd === undefined) ? 'None (First order)' : `${daysPrevOrd} days`;

    const daysPrevRet = bh.days_since_last_return;
    document.getElementById('bh-days-prev-ret').textContent = (daysPrevRet >= 900 || daysPrevRet === undefined) ? 'None (No returns)' : `${daysPrevRet} days`;

    const amtRatio = bh.amount_to_avg_ratio || 1.0;
    document.getElementById('bh-amount-ratio').textContent = `${amtRatio.toFixed(2)}×`;

    const refRatio = bh.refund_to_spend_ratio || 0.0;
    document.getElementById('bh-refund-spend-ratio').textContent = formatPct(refRatio);
  }

  // 5. Customer Timeline Renderer
  function renderTimeline(timeline) {
    const list = document.getElementById('customer-timeline-list');
    list.innerHTML = '';

    if (!timeline || timeline.length === 0) {
      list.innerHTML = '<div class="timeline-empty">Timeline information not available for this case.</div>';
      return;
    }

    timeline.forEach((step, idx) => {
      const item = document.createElement('div');
      item.className = `timeline-step step-type-${step.type || 'past'}`;

      let icon = '🔹';
      if (step.type === 'past') icon = '📜';
      if (step.type === 'current') icon = '📦';
      if (step.type === 'cutoff') icon = '🛑';
      if (step.type === 'eval') icon = '⚡';

      item.innerHTML = `
        <div class="timeline-node">
          <span class="timeline-icon">${icon}</span>
          ${idx < timeline.length - 1 ? '<div class="timeline-line"></div>' : ''}
        </div>
        <div class="timeline-body">
          <div class="timeline-header-row">
            <h5 class="timeline-stage-title">${step.stage}</h5>
            <span class="timeline-date-badge ${step.type === 'cutoff' ? 'badge-cutoff' : ''}">${step.date}</span>
          </div>
          <p class="timeline-desc">${step.description}</p>
        </div>
      `;
      list.appendChild(item);
    });
  }

  // 6. Entity Network & Topology Graph Renderer
  function renderEntityNetwork(network, customerId) {
    const dev = network.device || {};
    const ship = network.shipping_address || {};
    const bill = network.billing_address || {};
    const totalLinked = network.total_linked_external_accounts || 0;

    // Device Stats
    document.getElementById('ent-device-id').textContent = dev.device_id_masked || 'DEV-***';
    document.getElementById('ent-dev-accounts').textContent = dev.accounts_count ?? 1;
    document.getElementById('ent-dev-other-acc').textContent = dev.other_accounts_count ?? 0;
    document.getElementById('ent-dev-orders').textContent = dev.prior_orders_count ?? 0;
    document.getElementById('ent-dev-refunds').textContent = `${dev.prior_refunds_count ?? 0} (${formatINR(dev.prior_refund_amount || 0)})`;

    // Shipping Stats
    document.getElementById('ent-ship-id').textContent = ship.address_id_masked || 'ADDR-***';
    document.getElementById('ent-ship-accounts').textContent = ship.accounts_count ?? 1;
    document.getElementById('ent-ship-other-acc').textContent = ship.other_accounts_count ?? 0;
    document.getElementById('ent-ship-orders').textContent = ship.prior_orders_count ?? 0;
    document.getElementById('ent-ship-refunds').textContent = ship.prior_refunds_count ?? 0;

    // Billing Stats
    document.getElementById('ent-bill-id').textContent = bill.address_id_masked || 'ADDR-***';
    document.getElementById('ent-bill-accounts').textContent = bill.accounts_count ?? 1;
    document.getElementById('ent-bill-other-acc').textContent = bill.other_accounts_count ?? 0;
    document.getElementById('ent-bill-orders').textContent = bill.prior_orders_count ?? 0;
    document.getElementById('ent-bill-refunds').textContent = bill.prior_refunds_count ?? 0;

    document.getElementById('ent-total-linked').textContent = `Total Linked External Accounts: ${totalLinked}`;

    // Render Clean Relational SVG / Topology Tree
    renderEntityGraphDiagram(dev, ship, bill, customerId || 'CURRENT CUSTOMER', totalLinked);
  }

  function renderEntityGraphDiagram(dev, ship, bill, custId, totalLinked) {
    const container = document.getElementById('entity-graph-diagram');
    container.innerHTML = '';

    const devOtherAccs = dev.linked_accounts_anonymized || [];
    const shipOtherAccs = ship.linked_accounts_anonymized || [];
    const billOtherAccs = bill.linked_accounts_anonymized || [];

    // Construct diagram DOM
    const diagramHtml = `
      <div class="topology-wrapper">
        <!-- Top: Device Node -->
        <div class="topology-level level-top">
          <div class="topology-node node-device">
            <span class="tn-icon">💻</span>
            <div class="tn-meta">
              <span class="tn-type">DEVICE HARDWARE</span>
              <strong class="tn-name">${dev.device_id_masked || 'DEV-***'}</strong>
              <span class="tn-sub">${dev.accounts_count || 1} accounts · ${dev.prior_refunds_count || 0} refunds</span>
            </div>
          </div>
          ${devOtherAccs.length > 0 ? `
            <div class="linked-acc-pills-row">
              <span class="lap-label">Device Linked:</span>
              ${devOtherAccs.map(acc => `<span class="lap-tag">${acc}</span>`).join('')}
            </div>
          ` : '<span class="topology-single-tag">Single account footprint</span>'}
        </div>

        <!-- Trunk Connector -->
        <div class="topology-branch-vertical"></div>

        <!-- Center: Current Customer Node -->
        <div class="topology-level level-center">
          <div class="topology-node node-customer">
            <span class="tn-icon">👤</span>
            <div class="tn-meta">
              <span class="tn-type">EVALUATED CUSTOMER</span>
              <strong class="tn-name">${custId}</strong>
              <span class="tn-sub">Primary Dispute Account</span>
            </div>
          </div>
        </div>

        <!-- Lower Branch Fork -->
        <div class="topology-branch-fork">
          <div class="fork-line fork-left"></div>
          <div class="fork-line fork-right"></div>
        </div>

        <!-- Bottom: Addresses Level -->
        <div class="topology-level level-bottom">
          <!-- Shipping Address -->
          <div class="topology-node node-address">
            <span class="tn-icon">📍</span>
            <div class="tn-meta">
              <span class="tn-type">SHIPPING ADDRESS</span>
              <strong class="tn-name">${ship.address_id_masked || 'ADDR-***'}</strong>
              <span class="tn-sub">${ship.accounts_count || 1} account${(ship.accounts_count || 1) > 1 ? 's' : ''} · ${ship.prior_refunds_count || 0} refunds</span>
            </div>
            ${shipOtherAccs.length > 0 ? `
              <div class="node-linked-mini">
                ${shipOtherAccs.map(acc => `<span class="mini-tag">${acc}</span>`).join('')}
              </div>
            ` : ''}
          </div>

          <!-- Billing Address -->
          <div class="topology-node node-address">
            <span class="tn-icon">💳</span>
            <div class="tn-meta">
              <span class="tn-type">BILLING ADDRESS</span>
              <strong class="tn-name">${bill.address_id_masked || 'ADDR-***'}</strong>
              <span class="tn-sub">${bill.accounts_count || 1} account${(bill.accounts_count || 1) > 1 ? 's' : ''} · ${bill.prior_refunds_count || 0} refunds</span>
            </div>
            ${billOtherAccs.length > 0 ? `
              <div class="node-linked-mini">
                ${billOtherAccs.map(acc => `<span class="mini-tag">${acc}</span>`).join('')}
              </div>
            ` : ''}
          </div>
        </div>
      </div>
    `;

    container.innerHTML = diagramHtml;
  }

  // 7. What Stands Out Renderer
  function renderWhatStandsOut(standsOutList) {
    const list = document.getElementById('stands-out-items');
    list.innerHTML = '';

    if (!standsOutList || standsOutList.length === 0) {
      standsOutList = ["Customer transaction metrics align with legitimate historical baseline."];
    }

    standsOutList.forEach(text => {
      const li = document.createElement('li');
      li.className = 'stands-out-item';
      li.innerHTML = `
        <span class="so-bullet">⚡</span>
        <span class="so-text">${text}</span>
      `;
      list.appendChild(li);
    });
  }

  // 8. Why This Assessment (Structured Signals) Renderer
  function renderStructuredSignals(signals, fallbackTopSignals) {
    const container = document.getElementById('structured-signals-list');
    container.innerHTML = '';

    if (signals && signals.length > 0) {
      signals.forEach(sig => {
        const card = document.createElement('div');
        card.className = 'signal-card';
        card.innerHTML = `
          <div class="signal-card-header">
            <span class="sig-badge">Signal</span>
            <h5 class="sig-title">${sig.signal}</h5>
          </div>
          <div class="sig-body">
            <div class="sig-row">
              <span class="sig-label">Evidence:</span>
              <span class="sig-evidence">${sig.evidence}</span>
            </div>
            <div class="sig-row">
              <span class="sig-label">Interpretation:</span>
              <span class="sig-interp">${sig.interpretation}</span>
            </div>
          </div>
        `;
        container.appendChild(card);
      });
    } else if (fallbackTopSignals && fallbackTopSignals.length > 0) {
      fallbackTopSignals.forEach(item => {
        const card = document.createElement('div');
        card.className = 'signal-card';
        card.innerHTML = `
          <div class="signal-card-header">
            <span class="sig-badge">Signal</span>
            <h5 class="sig-title">Model C Key Indicator</h5>
          </div>
          <div class="sig-body">
            <div class="sig-row">
              <span class="sig-label">Evidence:</span>
              <span class="sig-evidence">${item}</span>
            </div>
            <div class="sig-row">
              <span class="sig-label">Interpretation:</span>
              <span class="sig-interp">Observed metric contributes to calibrated Model C risk score.</span>
            </div>
          </div>
        `;
        container.appendChild(card);
      });
    } else {
      container.innerHTML = '<div class="signal-empty">Observed metrics within standard consumer profile.</div>';
    }
  }

  // 9. Decision Support Renderer
  function renderDecisionSupport(ds, tier) {
    const card = document.getElementById('decision-support-card');
    const tierTag = document.getElementById('ds-tier-tag');
    const title = document.getElementById('ds-action-title');
    const guidance = document.getElementById('ds-guidance-text');
    const checklist = document.getElementById('ds-checklist-items');

    card.className = `intel-card decision-support-card ds-tier-${tier.toLowerCase()}`;
    tierTag.textContent = `${tier} TIER`;
    title.textContent = ds.action_title || (tier === 'LOW' ? 'NORMAL REFUND PROCESSING' : tier === 'MEDIUM' ? 'ADDITIONAL VERIFICATION' : 'HUMAN SPECIALIST REVIEW');
    guidance.textContent = ds.guidance || 'Review case context and proceed in accordance with operational refund routing protocols.';

    checklist.innerHTML = '';
    let steps = [];

    if (tier === 'LOW') {
      steps = [
        "Confirm item return dispatch status with carrier.",
        "Authorize standard refund disbursement under automatic approval rules.",
        "Log dispute resolution to audit repository."
      ];
    } else if (tier === 'MEDIUM') {
      steps = [
        "Review device hardware refund history and previous claims.",
        "Cross-reference courier delivery confirmation and package weight.",
        "Request customer verification photo of received item condition before disburse.",
        "Avoid automatic rejection; apply calibrated verification."
      ];
    } else {
      steps = [
        "Escalate claim to human fraud & abuse specialist team.",
        "Perform comprehensive multi-account entity network correlation analysis.",
        "Inspect shipping/billing address sharing across linked account identities.",
        "Human specialist makes final claim disposition (Automated denial is disabled)."
      ];
    }

    steps.forEach(step => {
      const li = document.createElement('li');
      li.textContent = step;
      checklist.appendChild(li);
    });
  }

});
