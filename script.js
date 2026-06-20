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


// Professional cursor-follow pop-up magnifier for large scientific figures.
// It shows a clipped enlarged copy of the figure and tracks the exact cursor position.
(function () {
  const zoomBox = document.getElementById('cursorZoom');
  const zoomWindow = document.getElementById('cursorZoomWindow');
  const zoomImg = document.getElementById('cursorZoomImage');
  const zoomTitle = document.getElementById('cursorZoomTitle');
  const images = Array.from(document.querySelectorAll('img[data-zoom="true"]'));
  if (!zoomBox || !zoomWindow || !zoomImg || !images.length) return;

  const crosshair = document.createElement('div');
  crosshair.className = 'cursor-zoom-crosshair';
  document.body.appendChild(crosshair);

  let activeImage = null;
  let lastEvent = null;
  let raf = null;
  const zoomScale = 2.75;

  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

  function isSmallScreen() {
    return window.matchMedia('(max-width: 900px), (pointer: coarse)').matches;
  }

  function placePreview(event) {
    const margin = 22;
    const boxW = zoomBox.offsetWidth || 600;
    const boxH = zoomBox.offsetHeight || 430;
    let x = event.clientX + boxW * 0.58;
    let y = event.clientY + boxH * 0.08;

    if (x + boxW / 2 > window.innerWidth - margin) x = event.clientX - boxW * 0.58;
    if (x - boxW / 2 < margin) x = margin + boxW / 2;
    if (y + boxH / 2 > window.innerHeight - margin) y = window.innerHeight - margin - boxH / 2;
    if (y - boxH / 2 < margin) y = margin + boxH / 2;

    zoomBox.style.left = `${x}px`;
    zoomBox.style.top = `${y}px`;
    crosshair.style.left = `${event.clientX}px`;
    crosshair.style.top = `${event.clientY}px`;
  }

  function updateMagnifier() {
    raf = null;
    if (!activeImage || !lastEvent) return;

    const rect = activeImage.getBoundingClientRect();
    const localX = clamp(lastEvent.clientX - rect.left, 0, rect.width);
    const localY = clamp(lastEvent.clientY - rect.top, 0, rect.height);
    const xRatio = rect.width ? localX / rect.width : 0.5;
    const yRatio = rect.height ? localY / rect.height : 0.5;

    const windowW = zoomWindow.clientWidth;
    const windowH = zoomWindow.clientHeight;
    const scaledW = rect.width * zoomScale;
    const scaledH = rect.height * zoomScale;

    zoomImg.style.width = `${scaledW}px`;
    zoomImg.style.height = `${scaledH}px`;

    const translateX = clamp(windowW / 2 - xRatio * scaledW, windowW - scaledW, 0);
    const translateY = clamp(windowH / 2 - yRatio * scaledH, windowH - scaledH, 0);
    zoomImg.style.transform = `translate3d(${translateX}px, ${translateY}px, 0)`;

    placePreview(lastEvent);
  }

  function requestUpdate(event) {
    lastEvent = event;
    if (!raf) raf = requestAnimationFrame(updateMagnifier);
  }

  function enterFigure(event) {
    if (isSmallScreen()) return;
    activeImage = event.currentTarget;
    zoomTitle.textContent = activeImage.dataset.zoomTitle || activeImage.alt || 'Figure magnifier';
    zoomImg.src = activeImage.currentSrc || activeImage.src;
    zoomBox.classList.add('active');
    crosshair.classList.add('active');
    requestUpdate(event);
  }

  function leaveFigure() {
    activeImage = null;
    lastEvent = null;
    zoomBox.classList.remove('active');
    crosshair.classList.remove('active');
    if (raf) cancelAnimationFrame(raf);
    raf = null;
  }

  images.forEach(img => {
    img.addEventListener('mouseenter', enterFigure);
    img.addEventListener('mousemove', requestUpdate);
    img.addEventListener('mouseleave', leaveFigure);
    img.addEventListener('click', () => {
      if (isSmallScreen()) window.open(img.currentSrc || img.src, '_blank', 'noopener');
    });
  });

  window.addEventListener('scroll', leaveFigure, { passive: true });
  window.addEventListener('resize', leaveFigure);
})();
