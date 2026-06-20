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

// High-resolution figure zoom viewer with scroll zoom and drag-to-pan
const imageZoomModal = document.getElementById('imageZoomModal');
const zoomImage = document.getElementById('zoomImage');
const zoomTitle = document.getElementById('zoomTitle');
const zoomStage = document.getElementById('zoomStage');
let zoomState = { scale: 1, x: 0, y: 0, dragging: false, startX: 0, startY: 0, originX: 0, originY: 0 };

function applyImageZoom() {
  if (!zoomImage) return;
  zoomImage.style.transform = `translate(calc(-50% + ${zoomState.x}px), calc(-50% + ${zoomState.y}px)) scale(${zoomState.scale})`;
}

function resetImageZoom() {
  zoomState.scale = 1;
  zoomState.x = 0;
  zoomState.y = 0;
  applyImageZoom();
}

function openImageZoom(src, title) {
  if (!imageZoomModal || !zoomImage) return;
  zoomImage.src = src;
  zoomTitle.textContent = title || 'Expanded figure';
  imageZoomModal.classList.add('active');
  imageZoomModal.setAttribute('aria-hidden', 'false');
  document.body.classList.add('zoom-lock');
  resetImageZoom();
}

function closeImageZoom() {
  if (!imageZoomModal || !zoomImage) return;
  imageZoomModal.classList.remove('active');
  imageZoomModal.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('zoom-lock');
  zoomImage.src = '';
}

function changeImageZoom(delta) {
  zoomState.scale = Math.min(4.5, Math.max(0.75, zoomState.scale + delta));
  applyImageZoom();
}

document.querySelectorAll('[data-zoom-src]').forEach(trigger => {
  trigger.addEventListener('click', () => openImageZoom(trigger.dataset.zoomSrc, trigger.dataset.zoomTitle));
});

document.querySelectorAll('[data-close-zoom]').forEach(button => button.addEventListener('click', closeImageZoom));
document.querySelector('[data-zoom-in]')?.addEventListener('click', () => changeImageZoom(0.25));
document.querySelector('[data-zoom-out]')?.addEventListener('click', () => changeImageZoom(-0.25));
document.querySelector('[data-zoom-reset]')?.addEventListener('click', resetImageZoom);

zoomStage?.addEventListener('wheel', event => {
  event.preventDefault();
  changeImageZoom(event.deltaY < 0 ? 0.18 : -0.18);
}, { passive: false });

zoomStage?.addEventListener('pointerdown', event => {
  zoomState.dragging = true;
  zoomState.startX = event.clientX;
  zoomState.startY = event.clientY;
  zoomState.originX = zoomState.x;
  zoomState.originY = zoomState.y;
  zoomStage.classList.add('dragging');
  zoomStage.setPointerCapture(event.pointerId);
});

zoomStage?.addEventListener('pointermove', event => {
  if (!zoomState.dragging) return;
  zoomState.x = zoomState.originX + (event.clientX - zoomState.startX);
  zoomState.y = zoomState.originY + (event.clientY - zoomState.startY);
  applyImageZoom();
});

zoomStage?.addEventListener('pointerup', event => {
  zoomState.dragging = false;
  zoomStage.classList.remove('dragging');
  try { zoomStage.releasePointerCapture(event.pointerId); } catch (_) {}
});

zoomStage?.addEventListener('pointercancel', () => {
  zoomState.dragging = false;
  zoomStage.classList.remove('dragging');
});

document.addEventListener('keydown', event => {
  if (event.key === 'Escape') closeImageZoom();
  if (!imageZoomModal?.classList.contains('active')) return;
  if (event.key === '+') changeImageZoom(0.25);
  if (event.key === '-') changeImageZoom(-0.25);
  if (event.key === '0') resetImageZoom();
});
