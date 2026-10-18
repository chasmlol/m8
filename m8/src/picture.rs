//! The player's view, as the game sends it for the mate's eyes: numbered spots drawn on it
//! (for the LLM to pick one), and written as a PNG (in base64, for the LLM's request).

use serde_json::Value;

/// Digits, 3 by 5 (a row a byte, its low 3 bits, left to right).
const DIGITS: [[u8; 5]; 10] = [
    [7, 5, 5, 5, 7],
    [2, 6, 2, 2, 7],
    [7, 1, 7, 4, 7],
    [7, 1, 7, 1, 7],
    [5, 5, 7, 1, 1],
    [7, 4, 7, 1, 7],
    [7, 4, 7, 5, 7],
    [7, 1, 1, 1, 1],
    [7, 5, 7, 5, 7],
    [7, 5, 7, 1, 7],
];
/// The digits' size, and the discs' they're on.
const SCALE: i64 = 3;
const RADIUS: i64 = 15;

/// Draws each spot (`id`; `x`, `y` from 0 to 1 across and down) as its number in white on a
/// dark disc with a white rim, centred on it.
pub fn mark(rgb: &mut [u8], width: usize, height: usize, spots: &[Value]) {
    let mut put = |x: i64, y: i64, colour: [u8; 3], alpha: f32| {
        if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
            let at = (y as usize * width + x as usize) * 3;
            for c in 0..3 {
                rgb[at + c] = (rgb[at + c] as f32 * (1.0 - alpha) + colour[c] as f32 * alpha) as u8;
            }
        }
    };
    for spot in spots {
        let (Some(id), Some(x), Some(y)) = (spot["id"].as_u64(), spot["x"].as_f64(), spot["y"].as_f64()) else {
            continue;
        };
        let (cx, cy) = ((x * width as f64) as i64, (y * height as f64) as i64);
        for dy in -RADIUS..=RADIUS {
            for dx in -RADIUS..=RADIUS {
                let distance = ((dx * dx + dy * dy) as f64).sqrt();
                if distance <= RADIUS as f64 - 2.0 {
                    put(cx + dx, cy + dy, [0, 0, 0], 0.7);
                } else if distance <= RADIUS as f64 {
                    put(cx + dx, cy + dy, [255, 255, 255], 1.0);
                }
            }
        }
        let text = id.to_string();
        let text_width = text.len() as i64 * 4 * SCALE - SCALE;
        let (left, top) = (cx - text_width / 2, cy - 5 * SCALE / 2);
        for (n, digit) in text.bytes().enumerate() {
            let rows = DIGITS[(digit - b'0') as usize];
            for (row, bits) in rows.iter().enumerate() {
                for column in 0..3 {
                    if bits >> (2 - column) & 1 == 1 {
                        for sy in 0..SCALE {
                            for sx in 0..SCALE {
                                put(
                                    left + n as i64 * 4 * SCALE + column * SCALE + sx,
                                    top + row as i64 * SCALE + sy,
                                    [255, 255, 255],
                                    1.0,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// A PNG of RGB rows from the top, stored without compression (it only goes to the LLM, on
/// this machine).
pub fn png(rgb: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut raw = Vec::with_capacity((width * 3 + 1) * height);
    for row in rgb.chunks(width * 3).take(height) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    // zlib: stored blocks of up to 65535 bytes, then the Adler-32 of the data
    let mut zlib = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65535).collect();
    for (n, block) in blocks.iter().enumerate() {
        zlib.push((n + 1 == blocks.len()) as u8);
        let length = block.len() as u16;
        zlib.extend_from_slice(&length.to_le_bytes());
        zlib.extend_from_slice(&(!length).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }
    zlib.extend_from_slice(&(b << 16 | a).to_be_bytes());

    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let start = out.len();
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let crc = crc32(&out[start..]);
        out.extend_from_slice(&crc.to_be_bytes());
    };
    let mut header = Vec::new();
    header.extend_from_slice(&(width as u32).to_be_bytes());
    header.extend_from_slice(&(height as u32).to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
    chunk(b"IHDR", &header);
    chunk(b"IDAT", &zlib);
    chunk(b"IEND", &[]);
    out
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 == 1 { 0xEDB8_8320 ^ (crc >> 1) } else { crc >> 1 };
        }
    }
    !crc
}

pub fn base64(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let n = (group[0] as u32) << 16 | (*group.get(1).unwrap_or(&0) as u32) << 8 | *group.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            out.push(if i <= group.len() { TABLE[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}
