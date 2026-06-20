const progress = document.querySelector('.scroll-progress');
const navToggle = document.querySelector('.nav-toggle');
const navLinks = document.querySelector('.nav-links');
const links = document.querySelectorAll('.nav-links a');

function updateProgress(){
  const scrollTop = window.scrollY || document.documentElement.scrollTop;
  const height = document.documentElement.scrollHeight - window.innerHeight;
  progress.style.width = `${height > 0 ? (scrollTop / height) * 100 : 0}%`;
}
window.addEventListener('scroll', updateProgress, { passive:true });
updateProgress();

navToggle?.addEventListener('click', () => {
  const isOpen = navLinks.classList.toggle('open');
  navToggle.setAttribute('aria-expanded', String(isOpen));
});

links.forEach(link => {
  link.addEventListener('click', () => {
    navLinks.classList.remove('open');
    navToggle?.setAttribute('aria-expanded', 'false');
  });
});

const revealObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if(entry.isIntersecting){
      entry.target.classList.add('visible');
      revealObserver.unobserve(entry.target);
    }
  });
}, { threshold:0.14 });
document.querySelectorAll('.reveal').forEach(el => revealObserver.observe(el));

const sections = [...document.querySelectorAll('main section[id], header[id]')];
const navObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if(entry.isIntersecting){
      links.forEach(a => a.classList.toggle('active', a.getAttribute('href') === `#${entry.target.id}`));
    }
  });
}, { rootMargin:'-45% 0px -50% 0px', threshold:0 });
sections.forEach(sec => navObserver.observe(sec));

function animateCounter(el){
  const target = parseFloat(el.dataset.target || '0');
  const duration = 1300;
  const start = performance.now();
  function step(now){
    const progress = Math.min((now - start) / duration, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = (target * eased).toFixed(3);
    if(progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

const counterObserver = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if(entry.isIntersecting){
      animateCounter(entry.target);
      counterObserver.unobserve(entry.target);
    }
  });
}, { threshold:0.55 });
document.querySelectorAll('.counter').forEach(counter => counterObserver.observe(counter));

const videoModal = document.getElementById('videoModal');
const modalVideo = document.getElementById('modalVideo');
document.querySelectorAll('[data-open-video]').forEach(btn => {
  btn.addEventListener('click', () => {
    videoModal.classList.add('open');
    videoModal.setAttribute('aria-hidden','false');
    modalVideo.currentTime = 0;
    modalVideo.play().catch(() => {});
  });
});
document.querySelectorAll('[data-close-modal]').forEach(btn => {
  btn.addEventListener('click', () => {
    videoModal.classList.remove('open');
    videoModal.setAttribute('aria-hidden','true');
    modalVideo.pause();
  });
});

const lightbox = document.getElementById('lightbox');
const lightboxImage = document.getElementById('lightboxImage');
document.querySelectorAll('[data-lightbox]').forEach(btn => {
  btn.addEventListener('click', () => {
    lightboxImage.src = btn.dataset.lightbox;
    lightbox.classList.add('open');
    lightbox.setAttribute('aria-hidden','false');
  });
});
document.querySelectorAll('[data-close-lightbox]').forEach(btn => {
  btn.addEventListener('click', () => {
    lightbox.classList.remove('open');
    lightbox.setAttribute('aria-hidden','true');
    lightboxImage.src = '';
  });
});

document.addEventListener('keydown', (event) => {
  if(event.key === 'Escape'){
    videoModal.classList.remove('open');
    videoModal.setAttribute('aria-hidden','true');
    modalVideo.pause();
    lightbox.classList.remove('open');
    lightbox.setAttribute('aria-hidden','true');
    lightboxImage.src = '';
  }
});
