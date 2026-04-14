/**
 * Image URL helper for CDN / local switching.
 *
 * Set PUBLIC_IMAGE_BASE_URL in .env to the CDN hostname
 * (e.g. https://francescoluchino-art.b-cdn.net) to serve from CDN.
 * Leave empty for local development (/images/... served from public/).
 */

const base = import.meta.env.PUBLIC_IMAGE_BASE_URL || "";

/**
 * Build a full image URL.
 * @param variant - e.g. "front/thumbs", "back/original"
 * @param filename - e.g. "{uuid}.webp"
 */
export function imageUrl(variant: string, filename: string): string {
  return `${base}/images/${variant}/${filename}`;
}
