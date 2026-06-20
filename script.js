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
