/**
 * masonry.js — Masonry layout + Intersection Observer lazy loading.
 *
 * Uses Desandro's masonry-layout for positioning and imagesloaded
 * to re-layout as images load. Lazy loading with blur-up is handled
 * via Intersection Observer.
 */

import Masonry from 'masonry-layout';
import imagesLoaded from 'imagesloaded';

(function () {
  'use strict';

  const COLUMN_WIDTH = 340;
  const GUTTER = 16; // matches --masonry-gap
  const ROOTMARGIN = '200px 0px';

  const gallery = document.querySelector('.gallery');
  if (!gallery) return;

  // --- Initialize Masonry ---
  const msnry = new Masonry(gallery, {
    itemSelector: '.painting-card',
    columnWidth: COLUMN_WIDTH,
    gutter: GUTTER,
    transitionDuration: '0.3s',
    initLayout: true,
  });

  // --- imagesLoaded: re-layout as images load ---
  imagesLoaded(gallery).on('progress', function () {
    msnry.layout();
  });

  // --- Intersection Observer lazy loading (blur-up) ---
  function loadImage(img) {
    const src = img.dataset.src;
    if (!src) return;

    const loader = new Image();
    loader.onload = function () {
      img.src = src;
      img.classList.remove('painting-card__img--placeholder');
      img.classList.add('painting-card__img--loaded');
    };
    loader.src = src;
    delete img.dataset.src;
  }

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
    document.querySelectorAll('.painting-card__img[data-src]').forEach((img) => {
      observer.observe(img);
    });
  } else {
    document.querySelectorAll('.painting-card__img[data-src]').forEach(loadImage);
  }

  // Expose instance for filter integration later
  window.__msnry = msnry;
})();
