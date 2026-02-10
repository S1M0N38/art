/**
 * masonry.js — Intersection Observer lazy loading with blur-up effect.
 *
 * CSS columns handle the masonry layout natively.
 * This script only manages image loading:
 *   1. Observe all .painting-card__img elements
 *   2. When a card enters the viewport buffer, swap placeholder → thumbnail
 *   3. On load, transition from blurred placeholder to sharp image
 */

(function () {
  'use strict';

  const ROOTMARGIN = '200px 0px'; // start loading 200px before visible

  /** Swap placeholder → real thumbnail and handle blur-up transition */
  function loadImage(img) {
    const src = img.dataset.src;
    if (!src) return;

    // Create a new Image to preload without flashing
    const loader = new Image();
    loader.onload = function () {
      img.src = src;
      img.classList.remove('painting-card__img--placeholder');
      img.classList.add('painting-card__img--loaded');
    };
    loader.src = src;

    // Clean up data attribute so we don't reload
    delete img.dataset.src;
  }

  // --- Intersection Observer ---
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(
      function (entries) {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            loadImage(entry.target);
            observer.unobserve(entry.target);
          }
        }
      },
      { rootMargin: ROOTMARGIN }
    );

    // Observe all painting images
    const images = document.querySelectorAll('.painting-card__img[data-src]');
    images.forEach(function (img) {
      observer.observe(img);
    });
  } else {
    // Fallback: load all images immediately
    document.querySelectorAll('.painting-card__img[data-src]').forEach(loadImage);
  }
})();
