package net_A;

import java.util.Random;

public class Cell {

	Random r = new Random(0);
	
	float[] h;
	float[] x;
	int t = 0;
	
	private float mod = .1f;
	private float wr = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float wz = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float wh = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float ur = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float uz = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float uh = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	
	float lr = 0.01f;
	
	public Cell() {
		
		float[] in = new float[10];
		float[] out = new float[10];
		
		for(int i = 0; i < 10000; i++) {
			for(int j = 0; j < 10; j++) {
				in[j] = r.nextFloat() * .9f;
				out[j] = func(in[j]);
			}
			backwardPass(in, out);
		}
		
		skip();
		
		while(true) {
			try {
				Thread.sleep(100);
				float i = r.nextFloat() * .9f;
				float o = forwardTimestep(i);
				System.out.println("N: "+t+" In: "+i+" Out: "+o+ " (Exp:"+func(i)+") -> err - "+(o - func(i)));
				if(t >= 10) {
					skip();
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	public float func(float x) {
		return sigm(x);
	}
	
	public float forwardTimestep(float in) {
		t++;
		h = expand(h);
		x = expand(x);
		return forwardStep(in, t-1);
	}

	public float[] forwardPass(float[] in) {
		int T = in.length;
		float[] out = new float[T];
		h = new float[T];
		x = new float[T];
		for(t = 0; t < T; t++) {
			out[t] = forwardStep(in[t], t);
		}
		return out;
	}
	
	public void backwardPass(float[] in, float[] y) {
		int T = in.length;
		h = new float[T];
		x = new float[T];	
		for(t = 0; t < T; t++) {
			forwardStep(in[t], t);
		}
		float[] out = new float[8];
		float[] tot = new float[6];
		
		for(t = T-1; t >= 0; t--) {
			out = backwardStep(out[1], t, y[t]);
			
			for(int i = 2; i < out.length; i++) {
				tot[i-2] += out[i];
			}
		}
		ur += tot[0] * lr;
		uz += tot[1] * lr;
		uh += tot[2] * lr;
		wr += tot[3] * lr;
		wz += tot[4] * lr;
		wh += tot[5] * lr;
	}
	
	public float forwardStep(float x, int t) {
		this.x[t] = x;
		float gate_r = ReLU(wr * get(h, t-1) + ur * x);
		float gate_z = ReLU(wz * get(h, t-1) + uz * x);
		float gate_h = tanh(wh * gate_r * get(h, t-1) + uh * x);
		h[t] = (1 - gate_z) * gate_h + gate_z * get(h, t-1);
		return h[t];
	}
	
	public float[] backwardStep(float dw, int t, float y) {
		float gate_r = ReLU(wr * get(h, t-1) + ur * x[t]);
		float gate_z = ReLU(wz * get(h, t-1) + uz * x[t]);
		float gate_h = tanh(wh * gate_r * get(h, t-1) + uh * x[t]);
		
		float d0 = dw + (y - h[t]);
		float d1 = gate_z * d0;
		float d2 = get(h, t-1) * d0;
		float d3 = gate_h * d0;
		float d4 = -1 * d3;
		float d5 = d2 + d4;
		float d6 = (1 - gate_z) * d0;
		float d7 = d5 * dReLU(gate_z);
		float d8 = d6 * (1 - (gate_h * gate_h));
		float d9 = d8 * uh;
		float d10 = d8 * wh;
		float d11 = d7 * uz;
		float d12 = d7 * wz;
		float d14 = d10 * gate_r;
		float d15 = d10 * get(h, t-1);
		float d16 = d15 * dReLU(gate_r);
		float d13 = d16 * ur;
		float d17 = d16 * wr;
		
		float dx = d9 + d11 + d13;
		float dh = d12 + d14 + d1 + d17;
		float dur = x[t] * d16;
		float duz = x[t] * d7;
		float duh = x[t] * d8;
		float dwr = get(h, t-1) * d16;
		float dwz = get(h, t-1) * d7;
		float dwh = get(h, t-1) * gate_r * d8;
		
		return new float[] {dx, dh, dur, duz, duh, dwr, dwz, dwh};
	}
	
	public void skip() {
		t = 0;
		h = new float[t];
		x = new float[t];
	}
	
	private float[] expand(float[] in) {
		float[] out = new float[in.length+1];
		for(int i = 0; i < in.length; i++) {
			out[i] = in[i];
		}
		return out;
	}
	
	public float get(float[] array, int index) {
		int min = 0;
		int max = array.length;
		if(index > min && index < max) {
			return array[index];
		}
		return 0;
	}
	
	public float tanh(float x) {
	    return (float) Math.tanh(x);
	}
	public float dtanh(float x) {
	    return (float) (1 - Math.pow(2, tanh(x)));
	}
	
	public float sigm(float x) {
		return  (1f / (1f + (float)Math.exp(-x)));
	}
	public float dsigm(float x) {
		return x * (1 - x);
	}
	
	private float ReLU(float x) {
		return (x > 0)? x : 0;
	}
	private float dReLU(float x) {
		return (x > 0)? 1 : 0;
	}
	
	public static void main(String[] args) {
		new Cell();
	}
}
