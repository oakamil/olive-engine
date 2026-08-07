// Required Notice: Copyright (c) 2026 Omair Kamil
// See LICENSE file in root directory for license terms.

pub struct Stacker {
    pub enabled: bool,
    buffer: Vec<u16>,
    out_buffer: Vec<u8>,
    pub stacked_frames: u32,
    pub virtual_exposure_us: u32,
    width: usize,
    height: usize,
}

impl Stacker {
    pub fn new(width: usize, height: usize, enabled: bool) -> Self {
        let size = width * height;
        Self {
            enabled,
            buffer: vec![0; size],
            out_buffer: vec![0; size],
            stacked_frames: 0,
            virtual_exposure_us: 0,
            width,
            height,
        }
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0);
        self.stacked_frames = 0;
        self.virtual_exposure_us = 0;
    }

    pub fn resize_if_needed(&mut self, width: usize, height: usize) {
        if self.width != width || self.height != height {
            let size = width * height;
            self.buffer = vec![0; size];
            self.out_buffer = vec![0; size];
            self.width = width;
            self.height = height;
            self.reset();
        }
    }

    #[inline(always)]
    fn compute_pass<const N: u16>(image: &[u8], buffer: &mut [u16], out_buffer: &mut [u8]) {
        // 64-bit chunk optimizations (8 bytes at a time)
        let chunks_8 = image.chunks_exact(8);
        let rem_8 = chunks_8.remainder();
        let buf_chunks = buffer.chunks_exact_mut(8);
        let mut out_chunks = out_buffer.chunks_exact_mut(8);

        for (img_chunk, (buf_chunk, out_chunk)) in chunks_8.zip(buf_chunks.zip(&mut out_chunks)) {
            for i in 0..8 {
                let sum = buf_chunk[i].saturating_add(img_chunk[i] as u16);
                buf_chunk[i] = sum;
                // Because N is a const generic, the compiler optimizes this div into a reciprocal imul
                out_chunk[i] = (sum / N) as u8;
            }
        }

        // Handle remainder correctly by zipping with the correct slice
        let buf_rem = buffer.chunks_exact_mut(8).into_remainder();
        let out_rem = out_buffer.chunks_exact_mut(8).into_remainder();
        for (img_val, (buf_val, out_val)) in
            rem_8.iter().zip(buf_rem.iter_mut().zip(out_rem.iter_mut()))
        {
            let sum = buf_val.saturating_add(*img_val as u16);
            *buf_val = sum;
            *out_val = (sum / N) as u8;
        }
    }

    pub fn add_frame<'a>(&'a mut self, image: &'a [u8], exposure_us: u32) -> &'a [u8] {
        if !self.enabled {
            self.virtual_exposure_us = exposure_us;
            self.stacked_frames = 1;
            return image;
        }

        assert_eq!(image.len(), self.buffer.len(), "Image length mismatch!");
        self.resize_if_needed(self.width, self.height);
        self.stacked_frames += 1;
        self.virtual_exposure_us += exposure_us;

        let n = self.stacked_frames as u16;

        // Dispatch to a const generic version to eliminate runtime division overhead
        match n {
            1 => Self::compute_pass::<1>(image, &mut self.buffer, &mut self.out_buffer),
            2 => Self::compute_pass::<2>(image, &mut self.buffer, &mut self.out_buffer),
            3 => Self::compute_pass::<3>(image, &mut self.buffer, &mut self.out_buffer),
            4 => Self::compute_pass::<4>(image, &mut self.buffer, &mut self.out_buffer),
            5 => Self::compute_pass::<5>(image, &mut self.buffer, &mut self.out_buffer),
            6 => Self::compute_pass::<6>(image, &mut self.buffer, &mut self.out_buffer),
            7 => Self::compute_pass::<7>(image, &mut self.buffer, &mut self.out_buffer),
            8 => Self::compute_pass::<8>(image, &mut self.buffer, &mut self.out_buffer),
            9 => Self::compute_pass::<9>(image, &mut self.buffer, &mut self.out_buffer),
            10 => Self::compute_pass::<10>(image, &mut self.buffer, &mut self.out_buffer),
            _ => {
                let mut buf_iter = self.buffer.iter_mut();
                let mut out_iter = self.out_buffer.iter_mut();
                for img_val in image.iter() {
                    let buf_val = buf_iter.next().unwrap();
                    let out_val = out_iter.next().unwrap();
                    let sum = buf_val.saturating_add(*img_val as u16);
                    *buf_val = sum;
                    *out_val = (sum / n) as u8;
                }
            }
        }

        &self.out_buffer
    }
}
