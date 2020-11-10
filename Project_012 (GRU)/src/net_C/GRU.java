package net_C;

import java.util.Random;

import net_B.Cell;

public class GRU {

	Random r = new Random(0);
	
	private float[] h;
	private float[] x;
	private float lr = .01f;
	private float mod = .1f;
	
	private float wr = r.nextFloat() * mod;//(r.nextBoolean()? mod : -mod);
	private float wz = r.nextFloat() * mod;//(r.nextBoolean()? mod : -mod);
	private float wg = r.nextFloat() * mod;//(r.nextBoolean()? mod : -mod);
	private float ur = r.nextFloat() * mod;//(r.nextBoolean()? mod : -mod);
	private float uz = r.nextFloat() * mod;//(r.nextBoolean()? mod : -mod);
	private float ug = r.nextFloat() * mod;//(r.nextBoolean()? mod : -mod);
	
	public GRU() {
		float[] in;
		float[] out;
		
		for(int i = 0; i < 10000; i++) {
			in = new float[] {r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat(), r.nextFloat()};
			out = new float[] {in[0]/2, in[1]/2, in[2]/2, in[3]/2, in[4]/2, in[5]/2, in[6]/2, in[7]/2, in[8]/2, in[9]/2};
			forwardPass(in);
			backwardPass(out);
		}
		in = new float[] {.2f, .8f, .3f, .5f, .4f, .6f, .9f, .0f, .1f, 1f};
		out = forwardPass(in);
		float tot = 0;
		for(int i = 0; i < out.length; i++) {
			tot += Math.abs(out[i] - in[i]/2);
			System.out.println("In: "+in[i]+" Out: "+out[i]+ " (Exp:"+in[i]/2+") -> err - "+(out[i] - in[i]/2));
		}
		System.out.println("Err: "+tot);
	}
	
	public float[] forwardPass(float[] in) {
		int T = in.length;
		float[] out = new float[T];
		h = new float[T];
		x = new float[T];
		for(int t = 0; t < T; t++) {
			x[t] = in[t];
			float r = sigm(wr * get(h, t-1) + ur * x[t]);
			float z = sigm(wz * get(h, t-1) + uz * x[t]);
			float g = tanh(wg * r * get(h, t-1) + ug * x[t]);
			h[t] = z * g + (1 - z) * get(h, t-1);
			out[t] = h[t];
		}
		return out;
	}
	
	public void backwardPass(float[] y) {
		float dw = 0;
		float[] error = new float[6];	
		
		for(int t = y.length-1; t > 0; t--) {
			float r = sigm(wr * get(h, t-1) + ur * x[t]);
			float z = sigm(wz * get(h, t-1) + uz * x[t]);
			float g = tanh(wg * r * get(h, t-1) + ug * x[t]);
	
			float d1 = (h[t] - y[t]);
			float d2 = dw;
			float d3 = d1 + d2;
			float d4 = (1 - z) * d3;
			//float d5 = d3 * h[t-1];
			//float d6 = 1 - d5;
			float d7 = d3 * g;
			float d8 = d3 * z;
			float d9 = d7 + d8;
			float d10 = d8 * dtanh(g);
			float d11 = d9 * dsigm(z);
			//float d12 = d10 * wg;
			float d13 = d10 * ug;
			//float d14 = d11 * wz;
			float d15 = d11 * uz;
			//float d16 = d13 * h[t-1];
			float d17 = d13 * r;
			float d18 = d17 * dsigm(r);
			float d19 = d17 + d4;
			//float d20 = d18 * wr;
			float d21 = d18 * ur;
			float d22 = d21 + d15;
			
			float dh = d19 + d22;
			//float dx = d12 + d14 + d20;
			dw = dh;
			
			error[0] += h[t-1] * d10;
			error[1] += x[t] * d10;
			error[2] += h[t-1] * d11;
			error[3] += x[t] * d11;
			error[4] += h[t-1] * r * d18;
			error[5] += x[t] * d18;
		}
		
		ur += error[0] * lr;
		uz += error[1] * lr;
		ug += error[2] * lr;
		wr += error[3] * lr;
		wz += error[4] * lr;
		wg += error[5] * lr;
	}
	
	public float get(float[] array, int index) {
		return (index > 0)? ((index < array.length)? array[index] : 0) : 0;
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
	
	public static void main(String[] args) {
		new GRU();
	}
}
