const progressBar = document.getElementById('progressBar');
const navToggle = document.getElementById('navToggle');
const navLinks = document.getElementById('navLinks');
const videoModal = document.getElementById('videoModal');
const modalVideo = document.getElementById('modalVideo');

window.addEventListener('scroll', () => {
  const scrollTop = document.documentElement.scrollTop || document.body.scrollTop;
  const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
  progressBar.style.width = `${height ? (scrollTop / height) * 100 : 0}%`;
});

navToggle?.addEventListener('click', () => {
  const open = navLinks.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
});

document.querySelectorAll('.nav-links a').forEach(link => {
  link.addEventListener('click', () => navLinks.classList.remove('open'));
});

document.querySelectorAll('a[href^="#"]').forEach(anchor => {
  anchor.addEventListener('click', event => {
    const target = document.querySelector(anchor.getAttribute('href'));
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
});

const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) entry.target.classList.add('active');
  });
}, { threshold: 0.14 });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

const counterObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting || entry.target.dataset.done) return;
    entry.target.dataset.done = 'true';
    const target = Number(entry.target.dataset.target);
    const duration = 1200;
    const start = performance.now();
    function tick(now) {
      const progress = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      entry.target.textContent = (target * eased).toFixed(3);
      if (progress < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  });
}, { threshold: 0.35 });

document.querySelectorAll('.counter').forEach(counter => counterObserver.observe(counter));

function openVideo() {
  videoModal.classList.add('active');
  videoModal.setAttribute('aria-hidden', 'false');
  modalVideo.currentTime = 0;
  modalVideo.play().catch(() => {});
}
function closeVideo() {
  videoModal.classList.remove('active');
  videoModal.setAttribute('aria-hidden', 'true');
  modalVideo.pause();
}

document.querySelectorAll('[data-open-video]').forEach(button => button.addEventListener('click', openVideo));
document.querySelectorAll('[data-close-video]').forEach(button => button.addEventListener('click', closeVideo));
document.addEventListener('keydown', event => { if (event.key === 'Escape') closeVideo(); });

document.querySelector('[data-copy-citation]')?.addEventListener('click', async event => {
  const text = document.getElementById('citationText').innerText;
  try {
    await navigator.clipboard.writeText(text);
    event.currentTarget.textContent = 'Copied';
    setTimeout(() => { event.currentTarget.textContent = 'Copy BibTeX'; }, 1500);
  } catch (error) {
    event.currentTarget.textContent = 'Select text manually';
  }
});


// Cursor-follow popup magnifier for manuscript figures.
// Works by reading the cursor position inside each figure and using the
// original high-resolution image as a movable background in a floating preview.
(function () {
  const zoomBox = document.getElementById('cursorZoom');
  const zoomWindow = document.getElementById('cursorZoomWindow');
  const zoomTitle = document.getElementById('cursorZoomTitle');
  const images = document.querySelectorAll('img[data-zoom="true"]');
  if (!zoomBox || !zoomWindow || !images.length) return;

  let activeImage = null;
  let raf = null;
  let latestEvent = null;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function placeZoomBox(event) {
    const margin = 24;
    const boxW = zoomBox.offsetWidth || 520;
    const boxH = zoomBox.offsetHeight || 360;
    let x = event.clientX + boxW * 0.54;
    let y = event.clientY - boxH * 0.10;

    if (x + boxW / 2 > window.innerWidth - margin) x = event.clientX - boxW * 0.58;
    if (x - boxW / 2 < margin) x = margin + boxW / 2;
    if (y + boxH / 2 > window.innerHeight - margin) y = window.innerHeight - margin - boxH / 2;
    if (y - boxH / 2 < margin) y = margin + boxH / 2;

    zoomBox.style.left = x + 'px';
    zoomBox.style.top = y + 'px';
  }

  function updateZoom() {
    raf = null;
    if (!activeImage || !latestEvent) return;

    const rect = activeImage.getBoundingClientRect();
    const xPct = clamp(((latestEvent.clientX - rect.left) / rect.width) * 100, 0, 100);
    const yPct = clamp(((latestEvent.clientY - rect.top) / rect.height) * 100, 0, 100);

    placeZoomBox(latestEvent);
    zoomWindow.style.backgroundImage = `url("${activeImage.currentSrc || activeImage.src}")`;
    zoomWindow.style.backgroundPosition = `${xPct}% ${yPct}%`;
  }

  function onEnter(event) {
    // Disable hover magnifier on touch/narrow screens.
    if (window.matchMedia('(max-width: 900px)').matches) return;
    activeImage = event.currentTarget;
    latestEvent = event;
    zoomTitle.textContent = activeImage.dataset.zoomTitle || activeImage.alt || 'Figure zoom';
    zoomBox.classList.add('active');
    updateZoom();
  }

  function onMove(event) {
    if (!activeImage) return;
    latestEvent = event;
    if (!raf) raf = requestAnimationFrame(updateZoom);
  }

  function onLeave() {
    activeImage = null;
    latestEvent = null;
    zoomBox.classList.remove('active');
    zoomWindow.style.backgroundImage = 'none';
  }

  images.forEach(img => {
    img.addEventListener('mouseenter', onEnter);
    img.addEventListener('mousemove', onMove);
    img.addEventListener('mouseleave', onLeave);

    // Mobile fallback: open the high-resolution figure in a new tab on tap.
    img.addEventListener('click', () => {
      if (window.matchMedia('(max-width: 900px)').matches) {
        window.open(img.currentSrc || img.src, '_blank', 'noopener');
      }
    });
  });

  window.addEventListener('scroll', onLeave, { passive: true });
  window.addEventListener('resize', onLeave);
})();
