const navToggle = document.getElementById('navToggle');
const mainNav = document.getElementById('mainNav');
const pageProgress = document.getElementById('pageProgress');
const modal = document.getElementById('videoModal');
const modalVideo = document.getElementById('modalVideo');

navToggle?.addEventListener('click', () => {
  const open = mainNav.classList.toggle('is-open');
  navToggle.setAttribute('aria-expanded', String(open));
});

document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', () => {
    mainNav?.classList.remove('is-open');
    navToggle?.setAttribute('aria-expanded', 'false');
  });
});

window.addEventListener('scroll', () => {
  const height = document.documentElement.scrollHeight - window.innerHeight;
  const progress = height > 0 ? (window.scrollY / height) * 100 : 0;
  pageProgress.style.width = `${progress}%`;
});

const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      entry.target.classList.add('in-view');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold: 0.16 });

document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

function animateCounter(el) {
  const target = Number(el.dataset.target || 0);
  const duration = 1400;
  const start = performance.now();

  function tick(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (target * eased).toFixed(3);
    if (progress < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

const counterObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting && !entry.target.dataset.done) {
      entry.target.dataset.done = 'true';
      animateCounter(entry.target);
    }
  });
}, { threshold: 0.45 });

document.querySelectorAll('.counter').forEach(counter => counterObserver.observe(counter));

function openVideo() {
  modal.classList.add('is-open');
  modal.setAttribute('aria-hidden', 'false');
  document.body.classList.add('modal-open');
  modalVideo.currentTime = 0;
  modalVideo.play().catch(() => {});
}

function closeVideo() {
  modal.classList.remove('is-open');
  modal.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('modal-open');
  modalVideo.pause();
}

document.querySelectorAll('[data-open-video]').forEach(button => button.addEventListener('click', openVideo));
document.querySelectorAll('[data-close-video]').forEach(button => button.addEventListener('click', closeVideo));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && modal.classList.contains('is-open')) closeVideo();
});

const copyCitation = document.getElementById('copyCitation');
const bibtex = document.getElementById('bibtex');
copyCitation?.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(bibtex.textContent.trim());
    copyCitation.textContent = 'Copied';
    setTimeout(() => { copyCitation.textContent = 'Copy BibTeX'; }, 1400);
  } catch (error) {
    copyCitation.textContent = 'Select text';
  }
});
