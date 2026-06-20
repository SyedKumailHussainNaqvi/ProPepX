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


// Cursor-following image zoom preview for publication figures
const zoomPreview = document.getElementById('imageZoomPreview');
const zoomLabel = zoomPreview?.querySelector('.zoom-preview-label');

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

document.querySelectorAll('.cursor-zoom-img').forEach(img => {
  img.closest('.figure-card')?.classList.add('has-hover-zoom');

  img.addEventListener('mouseenter', () => {
    if (!zoomPreview || window.matchMedia('(max-width: 900px)').matches) return;
    const zoom = Number(img.dataset.zoom || 2.2);
    zoomPreview.classList.add('active');
    zoomPreview.setAttribute('aria-hidden', 'false');
    zoomPreview.style.backgroundImage = `url("${img.currentSrc || img.src}")`;
    zoomPreview.style.backgroundSize = `${img.naturalWidth * zoom}px ${img.naturalHeight * zoom}px`;
    if (zoomLabel) zoomLabel.textContent = 'Cursor zoom · publication figure detail';
  });

  img.addEventListener('mousemove', event => {
    if (!zoomPreview || !zoomPreview.classList.contains('active')) return;

    const rect = img.getBoundingClientRect();
    const previewW = zoomPreview.offsetWidth;
    const previewH = zoomPreview.offsetHeight;
    const gap = 26;

    const xRatio = clamp((event.clientX - rect.left) / rect.width, 0, 1);
    const yRatio = clamp((event.clientY - rect.top) / rect.height, 0, 1);

    const bgW = img.naturalWidth * Number(img.dataset.zoom || 2.2);
    const bgH = img.naturalHeight * Number(img.dataset.zoom || 2.2);
    const bgX = -clamp(xRatio * bgW - previewW / 2, 0, Math.max(bgW - previewW, 0));
    const bgY = -clamp(yRatio * bgH - previewH / 2, 0, Math.max(bgH - previewH, 0));

    let left = event.clientX + gap;
    let top = event.clientY - previewH / 2;
    if (left + previewW + 18 > window.innerWidth) left = event.clientX - previewW - gap;
    top = clamp(top, 18, window.innerHeight - previewH - 18);

    zoomPreview.style.left = `${left}px`;
    zoomPreview.style.top = `${top}px`;
    zoomPreview.style.backgroundPosition = `${bgX}px ${bgY}px`;
  });

  img.addEventListener('mouseleave', () => {
    if (!zoomPreview) return;
    zoomPreview.classList.remove('active');
    zoomPreview.setAttribute('aria-hidden', 'true');
  });
});
