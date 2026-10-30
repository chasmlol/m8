//! The mouth's shapes for what's said: FaceGen's 16 phonemes over time, which the games built on
//! FaceGen (Oblivion, Fallout 3 and New Vegas, Skyrim) morph a speaking head with. A game maps
//! the frames onto its own lip format (Oblivion's `.lip` is these 16 weights at 30 a second).
//!
//! When each word is said comes from speech to text on the voice's own audio (`Transcriber::words`);
//! each word's letters give its shapes, spread over its time (vowels longer than consonants); the
//! voice's loudness opens and closes the mouth, so a pause between words shuts it. Without the
//! words' times, the line's text is spread over where there's voice.

use crate::listen::Word;

/// FaceGen's phonemes, in the order its heads (and Oblivion's lip files) keep them.
pub const PHONEMES: [&str; 16] = ["Aah", "BigAah", "BMP", "ChJSh", "DST", "Eee", "Eh", "FV", "I", "K", "N", "Oh", "OohQ", "R", "Th", "W"];

const AAH: usize = 0;
const BIG_AAH: usize = 1;
const BMP: usize = 2;
const CH: usize = 3;
const DST: usize = 4;
const EEE: usize = 5;
const EH: usize = 6;
const FV: usize = 7;
const I: usize = 8;
const K: usize = 9;
const N: usize = 10;
const OH: usize = 11;
const OOH: usize = 12;
const R: usize = 13;
const TH: usize = 14;
const W: usize = 15;

/// A frame of the mouth: each phoneme's weight, 0 to 1.
pub type Frame = [f32; 16];

/// The mouth's frames, `fps` a second, for `samples` (mono, at `rate`) saying `words` (when each
/// is said), or with none, `text` spread over where the audio has voice.
pub fn frames(samples: &[f32], rate: u32, words: &[Word], text: &str, fps: f32) -> Vec<Frame> {
    let seconds = samples.len() as f32 / rate.max(1) as f32;
    let count = (seconds * fps).ceil() as usize;
    if count == 0 {
        return Vec::new();
    }
    let loudness = envelope(samples, rate, fps, count);
    let segments = if words.is_empty() { spread(text, &loudness, fps) } else { timed(words) };

    let mut out = vec![[0.0f32; 16]; count];
    for (f, frame) in out.iter_mut().enumerate() {
        let t = (f as f32 + 0.5) / fps;
        let Some(at) = segments.iter().position(|s| t >= s.1 && t < s.2) else { continue };
        let (shape, start, end) = segments[at];
        // Each shape blends into its neighbours over the outer third of its time (the mouth is
        // already on its way to the next one), and holds alone in the middle.
        let length = (end - start).max(1e-3);
        let into = (t - start) / length;
        let edge = 0.33;
        let mut put = |shape: usize, weight: f32| frame[shape] = frame[shape].max(weight);
        put(shape, 1.0);
        if into < edge {
            if let Some(&(before, _, before_end)) = at.checked_sub(1).and_then(|b| segments.get(b)) {
                if (start - before_end).abs() < 1e-3 {
                    put(before, 0.5 * (1.0 - into / edge));
                }
            }
        } else if into > 1.0 - edge {
            if let Some(&(after, after_start, _)) = segments.get(at + 1) {
                if (after_start - end).abs() < 1e-3 {
                    put(after, 0.5 * (into - (1.0 - edge)) / edge);
                }
            }
        }
        // The voice's loudness opens the mouth: vowels open with it (a loud open vowel is the big
        // one), consonants keep their shape while there's any voice, and silence shuts it.
        let open = loudness[f];
        for (p, weight) in frame.iter_mut().enumerate() {
            let vowel = matches!(p, AAH | BIG_AAH | EEE | EH | I | OH | OOH);
            *weight *= if vowel { 0.3 + 0.7 * open } else { (open * 4.0).min(1.0) };
        }
        if frame[AAH] > 0.0 && open > 0.75 {
            frame[BIG_AAH] = frame[BIG_AAH].max(frame[AAH]);
            frame[AAH] *= 0.4;
        }
    }
    smooth(&mut out);
    out
}

/// The shapes a word's letters make, in order: English spelling, read the way it's said often
/// enough for a mouth (the digraphs first, a silent final e, letters that make no shape of their
/// own left out).
pub fn shapes(word: &str) -> Vec<usize> {
    let letters: Vec<char> = word.to_lowercase().chars().filter(|c| c.is_ascii_alphabetic()).collect();
    let is_vowel = |c: char| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
    let mut out = Vec::new();
    let mut i = 0;
    while i < letters.len() {
        let rest: String = letters[i..].iter().take(3).collect();
        let next = letters.get(i + 1).copied();
        let (made, took): (&[usize], usize) = match () {
            _ if rest.starts_with("tch") => (&[CH], 3),
            _ if rest.starts_with("th") => (&[TH], 2),
            _ if rest.starts_with("sh") || rest.starts_with("ch") => (&[CH], 2),
            _ if rest.starts_with("ph") => (&[FV], 2),
            _ if rest.starts_with("wh") => (&[W], 2),
            _ if rest.starts_with("ck") => (&[K], 2),
            _ if rest.starts_with("ng") => (&[N], 2),
            _ if rest.starts_with("qu") => (&[K, W], 2),
            _ if rest.starts_with("kn") && i == 0 => (&[N], 2),
            _ if rest.starts_with("wr") && i == 0 => (&[R], 2),
            _ if rest.starts_with("gh") => (&[], 2),
            _ if rest.starts_with("oo") => (&[OOH], 2),
            _ if ["ee", "ea", "ie", "ey"].iter().any(|d| rest.starts_with(d)) => (&[EEE], 2),
            _ if ["ai", "ay", "ei"].iter().any(|d| rest.starts_with(d)) => (&[EH, EEE], 2),
            _ if rest.starts_with("ou") || rest.starts_with("ow") => (&[OH, W], 2),
            _ if rest.starts_with("oi") || rest.starts_with("oy") => (&[OH, EEE], 2),
            _ if rest.starts_with("au") || rest.starts_with("aw") => (&[OH], 2),
            _ if ["er", "ir", "ur"].iter().any(|d| rest.starts_with(d)) && !letters.get(i + 2).is_some_and(|&c| is_vowel(c)) => {
                (&[R], 2)
            }
            _ => {
                let c = letters[i];
                // (a final e after a consonant is silent where the word has a vowel before it:
                // "make", "stone", but not "the" or "be")
                let silent_e = c == 'e'
                    && i + 1 == letters.len()
                    && i >= 2
                    && !is_vowel(letters[i - 1])
                    && letters[..i - 1].iter().any(|&c| is_vowel(c));
                let soft_c = c == 'c' && next.is_some_and(|n| matches!(n, 'e' | 'i' | 'y'));
                let made: &[usize] = match c {
                    _ if silent_e => &[],
                    'a' => &[AAH],
                    'e' => &[EH],
                    'i' => &[I],
                    'o' => &[OH],
                    // (a short u, "but", is open; a long one, "rule", rounds the lips)
                    'u' => if next.is_some_and(|n| !is_vowel(n)) && letters.get(i + 2).is_none_or(|&c| !is_vowel(c)) { &[AAH] } else { &[OOH] },
                    'y' => if i == 0 { &[EEE] } else { &[I] },
                    'b' | 'm' | 'p' => &[BMP],
                    'f' | 'v' => &[FV],
                    'w' => &[W],
                    'r' => &[R],
                    'l' | 'n' => &[N],
                    'j' => &[CH],
                    'c' if soft_c => &[DST],
                    'c' | 'k' | 'g' | 'q' => &[K],
                    'd' | 't' | 's' | 'z' | 'x' => &[DST],
                    _ => &[],
                };
                (made, 1)
            }
        };
        // (a letter doubled says itself once: "ll", "ss")
        for &shape in made {
            if out.last() != Some(&shape) || matches!(shape, AAH | EH | I | OH | OOH | EEE) && took > 1 {
                out.push(shape);
            }
        }
        i += took;
    }
    out
}

/// How long a shape takes, against the others in its word: vowels are held, closures are quick.
fn weight(shape: usize) -> f32 {
    match shape {
        AAH | BIG_AAH | EEE | EH | I | OH | OOH => 2.0,
        BMP => 0.8,
        _ => 1.0,
    }
}

/// Each word's shapes over its own time: (shape, start, end) in seconds.
fn timed(words: &[Word]) -> Vec<(usize, f32, f32)> {
    let mut segments = Vec::new();
    for word in words {
        place(&shapes(&word.text), word.start, word.end, &mut segments);
    }
    segments
}

/// The shapes of `text`'s words over the voiced frames (where the loudness says there's voice),
/// each word a share of the voiced time by its letters.
fn spread(text: &str, loudness: &[f32], fps: f32) -> Vec<(usize, f32, f32)> {
    let voiced: Vec<usize> = (0..loudness.len()).filter(|&f| loudness[f] > 0.15).collect();
    let words: Vec<&str> = text.split_whitespace().filter(|w| w.chars().any(char::is_alphabetic)).collect();
    let letters: usize = words.iter().map(|w| w.chars().filter(char::is_ascii_alphabetic).count()).sum();
    let mut segments = Vec::new();
    if voiced.is_empty() || letters == 0 {
        return segments;
    }
    let mut done = 0usize;
    for word in words {
        let n = word.chars().filter(char::is_ascii_alphabetic).count();
        let from = voiced[(done * voiced.len() / letters).min(voiced.len() - 1)];
        done += n;
        let to = voiced[(done * voiced.len() / letters).min(voiced.len()) - 1] + 1;
        place(&shapes(word), from as f32 / fps, to as f32 / fps, &mut segments);
    }
    segments
}

/// Puts `shapes` one after another over `start`..`end`, by their weights.
fn place(shapes: &[usize], start: f32, end: f32, segments: &mut Vec<(usize, f32, f32)>) {
    let total: f32 = shapes.iter().map(|&s| weight(s)).sum();
    if total <= 0.0 || end <= start {
        return;
    }
    let mut at = start;
    for &shape in shapes {
        let length = (end - start) * weight(shape) / total;
        segments.push((shape, at, at + length));
        at += length;
    }
}

/// The voice's loudness at each frame, 0 (silence) to 1 (its loud parts): RMS over a frame and a
/// half, against the line's own loud parts, with quiet below a floor taken as silence.
fn envelope(samples: &[f32], rate: u32, fps: f32, count: usize) -> Vec<f32> {
    let hop = rate as f32 / fps;
    let half = (hop * 0.75) as usize;
    let rms: Vec<f32> = (0..count)
        .map(|f| {
            let centre = ((f as f32 + 0.5) * hop) as usize;
            let window = &samples[centre.saturating_sub(half).min(samples.len())..(centre + half).min(samples.len())];
            if window.is_empty() {
                return 0.0;
            }
            (window.iter().map(|s| s * s).sum::<f32>() / window.len() as f32).sqrt()
        })
        .collect();
    let mut sorted = rms.clone();
    sorted.sort_by(f32::total_cmp);
    let loud = sorted[(sorted.len() * 9 / 10).min(sorted.len() - 1)].max(1e-4);
    rms.iter().map(|&r| ((r / loud - 0.08) / 0.92).clamp(0.0, 1.0)).collect()
}

/// Each phoneme's weight over three frames (a quarter, a half, a quarter): a mouth doesn't jump
/// from one shape to the next in a thirtieth of a second.
fn smooth(frames: &mut [Frame]) {
    let original = frames.to_vec();
    for f in 0..frames.len() {
        for p in 0..16 {
            let at = |g: usize| original[g.min(original.len() - 1)][p];
            frames[f][p] = 0.25 * at(f.saturating_sub(1)) + 0.5 * at(f) + 0.25 * at(f + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn named(shapes: &[usize]) -> Vec<&'static str> {
        shapes.iter().map(|&s| PHONEMES[s]).collect()
    }

    #[test]
    fn words_make_their_shapes() {
        assert_eq!(named(&shapes("the")), ["Th", "Eh"]);
        assert_eq!(named(&shapes("make")), ["BMP", "Aah", "K"]);
        assert_eq!(named(&shapes("shop")), ["ChJSh", "Oh", "BMP"]);
        assert_eq!(named(&shapes("Well,")), ["W", "Eh", "N"]);
        assert_eq!(named(&shapes("queen")), ["K", "W", "Eee", "N"]);
        assert_eq!(named(&shapes("knight")), ["N", "I", "DST"]);
        assert_eq!(named(&shapes("but")), ["BMP", "Aah", "DST"]);
        assert_eq!(named(&shapes("rule")), ["R", "OohQ", "N"]);
        assert!(shapes("...").is_empty());
    }

    #[test]
    fn the_mouth_moves_only_while_it_speaks() {
        let rate = 16000;
        // (a second of silence, half a second of a tone standing in for a voice, silence again)
        let mut samples = vec![0.0f32; rate as usize];
        samples.extend((0..rate / 2).map(|i| 0.3 * (i as f32 * 0.05).sin()));
        samples.extend(vec![0.0f32; rate as usize]);
        let words = [Word { text: "hello".into(), start: 1.0, end: 1.5 }];
        let frames = frames(&samples, rate, &words, "hello", 30.0);
        assert_eq!(frames.len(), 75);
        let moving = |f: &Frame| f.iter().any(|&w| w > 0.05);
        assert!(!frames[..28].iter().any(moving), "shut before the word");
        assert!(frames[31..44].iter().all(moving), "moving through it");
        assert!(!frames[48..].iter().any(moving), "shut after it");
        // (with no word times, the text goes where the voice is)
        let spread = super::frames(&samples, rate, &[], "hello", 30.0);
        assert!(spread[31..44].iter().all(moving) && !spread[..28].iter().any(moving));
    }
}
