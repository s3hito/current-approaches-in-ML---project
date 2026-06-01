import { createServer } from 'http';
import { Beatmap, Difficulty } from 'rosu-pp-js'; // Performance no longer needed

const PORT = 3000;

const server = createServer((req, res) => {
    if (req.method !== 'POST') { res.writeHead(405); res.end(); return; }

    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', () => {
        let map = null;
        try {
            const { mapId, mods, content } = JSON.parse(body);

            // Beatmap accepts a string directly — no Uint8Array needed
            map = new Beatmap(Buffer.from(content, 'base64').toString('utf-8'));

            // Difficulty returns DifficultyAttributes with aim/speed as top-level fields
            // Mods can be passed as string array directly — no bitmask conversion needed
			const attrs = new Difficulty({ mods: mods || [] }).calculate(map);
			console.log('[DEBUG attrs]', JSON.stringify(attrs.toJSON())); // add this
            const result = {
                star_ratings: {
                    NM: {
                        star_rating:  attrs.stars,
                        aim_rating:   attrs.aim,
                        speed_rating: attrs.speed,
                    }
                }
            };

            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(result));
        } catch (e) {
            console.error('[SR ERROR]', e);
            res.writeHead(500, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: e.message, stack: e.stack }));
        } finally {
            map?.free(); // Always free to prevent memory leaks
        }
    });
});

server.listen(PORT, () => console.log(`SR server listening on port ${PORT}`));