%% experiments
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"3")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2];[6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"4")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2];[-6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"5")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[-6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"6")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[-6,-6];[5,7]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"7")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"fix")
%%
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"3fix")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2];[6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"4fix")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[1,2];[-6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"5fix")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[-6,-6]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"6fix")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0,3.0,5.0]);
xs = [[5,0];[-5,0];[0,6];[0,-5];[-6,-6];[5,7]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"7fix")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0,3.0,5.0]);
xs = [[5,0];[-5,0];[2,3];[-2,3];[-2,-3];[2,-3]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"8fix")
%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0,3.0,5.0]);
xs = [[5,0];[-5,0];[2,4];[-2,4];[-2,-4];[2,-4]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"9fix")

%%
sigma_seq = 0.2*exp([0.5,1,1.5,2.0,3.0,5.0]);
xs = [[5,0];[-5,0];[2,4];[-2,4];[-2,-4];[2,-4]];
compute_diffusionSpectra(xs,sigma_seq,10,201,"8rnd")
