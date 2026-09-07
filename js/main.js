// Main interactions: theme, reveal, title links.
const root = document.documentElement;
const toggle = document.getElementById('theme-toggle');
const metaTheme = document.querySelector('meta[name="theme-color"]');

const THEME_COLORS = { dark: '#12110e', light: '#f5f0e4' };

function applyTheme(next, persist = true) {
  root.setAttribute('data-theme', next);
  toggle?.setAttribute('aria-pressed', String(next === 'dark'));
  toggle?.setAttribute('aria-label', next === 'dark' ? 'Switch to light theme' : 'Switch to dark theme');
  if (metaTheme) metaTheme.setAttribute('content', THEME_COLORS[next]);
  if (persist) {
    try { localStorage.setItem('theme', next); } catch { /* private mode */ }
  }
}

function currentTheme() {
  return root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
}

// Sync toggle state with theme set in <head> (avoids FOUC mismatch).
applyTheme(currentTheme(), false);

toggle?.addEventListener('click', () => {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  // Smooth cross-fade where supported, instant otherwise.
  if (document.startViewTransition) {
    document.startViewTransition(() => applyTheme(next));
  } else {
    applyTheme(next);
  }
});

// Follow OS changes only when the user has no saved preference.
try {
  if (!localStorage.getItem('theme')) {
    window.matchMedia('(prefers-color-scheme: light)').addEventListener('change', (e) => {
      applyTheme(e.matches ? 'light' : 'dark', false);
    });
  }
} catch { /* ignore */ }

// --- Staggered scroll reveal ---
const revealEls = document.querySelectorAll('.reveal');
if ('IntersectionObserver' in window && revealEls.length) {
  // Stagger siblings inside the same parent for a cascading feel.
  const groups = new Map();
  revealEls.forEach((el) => {
    const parent = el.parentElement;
    if (!groups.has(parent)) groups.set(parent, []);
    groups.get(parent).push(el);
  });
  groups.forEach((items) => {
    if (items.length > 1) {
      items.forEach((el, i) => {
        if (i > 0 && i <= 5 && !el.classList.contains('reveal-delay-1')) {
          el.style.transitionDelay = `${Math.min(i * 60, 300)}ms`;
        }
      });
    }
  });

  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -30px 0px' });
  revealEls.forEach((el) => obs.observe(el));
} else {
  revealEls.forEach((el) => el.classList.add('visible'));
}

// --- Collapse older publications behind a toggle ---
const pubListEl = document.getElementById('pub-list');
if (pubListEl) {
  const older = [...pubListEl.querySelectorAll('.pub-item')].filter((i) => +i.dataset.year < 2022);
  if (older.length) {
    older.forEach((i) => i.classList.add('is-older'));
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'older-toggle';
    const update = () => {
      const open = pubListEl.classList.contains('show-older');
      btn.textContent = open ? 'Show fewer publications' : `Show older publications (${older.length})`;
      btn.setAttribute('aria-expanded', String(open));
    };
    btn.addEventListener('click', () => {
      pubListEl.classList.toggle('show-older');
      // Hidden rows never intersected, so reveal them on expand.
      older.forEach((i) => i.classList.add('visible'));
      update();
    });
    update();
    pubListEl.after(btn);
  }
}

// --- 3D tilt on the portrait (fine pointers only, no reduced motion) ---
(function initTilt() {
  const wrap = document.querySelector('.profile-photo');
  const img = wrap?.querySelector('img');
  if (!wrap || !img) return;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  if (!window.matchMedia('(pointer: fine)').matches) return;
  const MAX = 9;
  wrap.addEventListener('mousemove', (e) => {
    const r = wrap.getBoundingClientRect();
    const px = (e.clientX - r.left) / r.width - 0.5;
    const py = (e.clientY - r.top) / r.height - 0.5;
    img.classList.add('tilting');
    img.style.setProperty('--tilt-y', `${(px * MAX).toFixed(2)}deg`);
    img.style.setProperty('--tilt-x', `${(-py * MAX).toFixed(2)}deg`);
  });
  wrap.addEventListener('mouseleave', () => {
    img.classList.remove('tilting');
    img.style.removeProperty('--tilt-x');
    img.style.removeProperty('--tilt-y');
  });
})();

// --- Footer year ---
const yearEl = document.getElementById('year');
if (yearEl) yearEl.textContent = String(new Date().getFullYear());

// --- Linkify publication titles to their primary link (paper/project) ---
document.querySelectorAll('.pub-item').forEach((item) => {
  const title = item.querySelector('.pub-title');
  const primary = item.querySelector('.pub-meta a');
  if (!title || !primary || title.querySelector('a')) return;
  const link = document.createElement('a');
  link.href = primary.href;
  link.rel = 'noopener noreferrer';
  link.textContent = title.textContent;
  title.replaceChildren(link);
});
