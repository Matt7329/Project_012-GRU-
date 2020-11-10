package net_B;

import java.util.Random;

public class Cell {

	Random r = new Random(0);
	
	float[] h = new float[0];
	float[] x = new float[0];
	
	int t = 0;
	
	private float mod = .1f;
	private float wr = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float wz = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float wh = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float ur = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float uz = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	private float uh = r.nextFloat() * (r.nextBoolean()? mod : -mod);
	
	float lr = .01f;
	
	public Cell() {
		
		float[] in;
		float[] out;
		
		for(int i = 0; i < 10000; i++) {
			in = new float[] {r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat()};
			out = new float[] {func(in[0]), func(in[1]), func(in[2]), func(in[3]), func(in[4]), func(in[5]), func(in[6]), func(in[7]), func(in[8]), func(in[9])};
			
			for(int j = 0; j < in.length; j++) {
				forwardStep(in[j]);
			}
			backwardPass(out);
		}
		
		System.out.println(t);
		
		skip();
		
		int n = 0;
		while(true) {
			try {
				Thread.sleep(100);
				n++;
				float i = r.nextFloat();
				float o = forwardStep(i);
				System.out.println("N: "+n+" In: "+i+" Out: "+o+ " (Exp:"+func(i)+") -> err - "+(o - func(i)));
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}
	
	public float func(float x) {
		return x + .1f;
	}
	
	public void backwardPass(float[] y) {
		int T = t;
		float[] out = new float[8];
		float[] tot = new float[6];	
		while(t > T - y.length) {
			t--;
			int index = (y.length) - (T - t);
			out = backwardStep(out[1], y[index]);
			
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
	
	public float forwardStep(float in) {
		h = expand(h);
		x = expand(x);
		
		this.x[t] = in;
		float gate_r = ReLU(wr * get(h, t-1) + ur * x[t]);
		float gate_z = ReLU(wz * get(h, t-1) + uz * x[t]);
		float gate_h = tanh(wh * gate_r * get(h, t-1) + uh * x[t]);
		h[t] = (1 - gate_z) * gate_h + gate_z * get(h, t-1);
		
		t++;
		
		return h[t-1];
	}
	
	public float[] backwardStep(float dw, float y) {
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
		h = new float[0];
		x = new float[0];
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
