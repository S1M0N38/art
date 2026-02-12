/**
 * Critica page — expand/collapse reviews + scroll-reveal animation.
 */

// ── Expand / Collapse ──
document.querySelectorAll('.review-text__toggle').forEach((btn) => {
  const wrapper = btn.previousElementSibling;
  if (!wrapper) return;
  const content = wrapper.querySelector('.review-text__content');
  if (!content) return;

  // Hide button if text isn't actually clamped (short reviews)
  requestAnimationFrame(() => {
    if (content.scrollHeight <= content.clientHeight + 2) {
      btn.style.display = 'none';
    }
  });

  btn.addEventListener('click', () => {
    const expanded = wrapper.classList.toggle('expanded');
    btn.textContent = expanded ? 'Chiudi ‹' : 'Leggi tutto ›';
    btn.setAttribute('aria-expanded', String(expanded));
  });
});

// ── Scroll-reveal with stagger ──
const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.08, rootMargin: '0px 0px -40px 0px' },
);

document.querySelectorAll('.critic-card').forEach((card, i) => {
  card.style.transitionDelay = `${Math.min(i * 0.06, 0.3)}s`;
  observer.observe(card);
});
