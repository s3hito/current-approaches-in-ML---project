
import { calculateStarRating } from 'osu-sr-calculator';

// Get command-line arguments
const mapId = Number(process.argv[2]);

// Mods are optional
// Example:
// node sr.js 53 HD DT
const mods = process.argv.slice(3);

// Build options object
const options = {};

if (mods.length > 0) {
    options.mods = mods;
}

// Calculate SR
const result = await calculateStarRating(mapId, options);

// Output JSON for Python
console.log(JSON.stringify(result, null, 2));

