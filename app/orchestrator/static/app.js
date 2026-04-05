// NovelConverter – shared JS utilities

// Generic API helper
async function apiRequest(method, url, body = null) {
  const opts = {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : {},
  };
  if (body) opts.body = JSON.stringify(body);
  const res = await fetch(url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = err?.detail ?? err;
    const normalized = (detail && typeof detail === 'object')
      ? {
          reason: detail.reason || 'Request failed',
          code: detail.code || `http_${res.status}`,
          detail: detail.detail ?? detail,
        }
      : {
          reason: String(detail || res.statusText || 'Request failed'),
          code: `http_${res.status}`,
          detail: String(detail || ''),
        };
    const e = new Error(`${normalized.reason} [${normalized.code}]`);
    e.reason = normalized.reason;
    e.code = normalized.code;
    e.detail = normalized.detail;
    throw e;
  }
  return res.json();
}

// Flash message utility
function flash(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `save-status ${type === 'error' ? 'error' : type === 'success' ? 'success' : ''}`;
  el.textContent = msg;
  el.style.position = 'fixed';
  el.style.bottom = '24px';
  el.style.right = '24px';
  el.style.zIndex = '9999';
  el.style.maxWidth = '400px';
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4000);
}
