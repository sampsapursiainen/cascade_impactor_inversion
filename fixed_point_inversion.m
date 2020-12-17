%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Copyright @ 2019, Laura Valtonen, Sampo Saari and Sampsa Pursiainen
%
%This script demonstrates inverting cascade impactor measurements via 
%a fixed-point and L1-regularized iterative alternating sequential (IAS) 
%approach. 
%
%For plotting the results this script requires the IoSR Matlab Toolbox 
%(Copyright (c) 2016 Institute of Sound Recording) available at:
%https://github.com/IoSR-Surrey/MatlabToolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
import iosr.statistics.*

%Choose the kernel type between actual (kernel_type = 1) and synthetic
%(kernel_type = 2).
kernel_type = 1;

%Set other parameter values.
L1_reg_param=1e-8;
L1_eps_val = 1e-4;
sample_size = 100;
noise = [0.01:0.005:0.03];
n_iterations = 20;
n_kernels = 12;
n_test = 6;

time_fp_ker = zeros(length(noise),sample_size,n_test);
time_fp_pnt = zeros(length(noise),sample_size,n_test);
time_L1_ker = zeros(length(noise),sample_size,n_test);
time_L1_pnt = zeros(length(noise),sample_size,n_test);

%Import kernels
if kernel_type == 1 
load elpiker.mat
s = log(elpiker.dp);
s_exp = elpiker.dp;
s_1 = min(s);
s_2 = max(s);
kernel_cell = elpiker.ker1;
for i = 1 : n_kernels
f(i,:) = kernel_cell{i};
%Conversion to mass concentration
f(i,:) = f(i,:).*s_exp.^3*1/6*pi();
end
else

%Set s-axis
s_res = 500;
s_1 = log(0.003);
s_2 = log(10);
d_s = (s_2-s_1)/s_res; 
s = [s_1:d_s:s_2];
s_exp = exp(s);

f = zeros(n_kernels, size(s,2));
var_basis = (s_2-s_1)/(1.2*n_kernels);
f_peaks = [s_1+(s_2-s_1)/(n_kernels+2):(s_2-s_1)/(n_kernels+2):s_2-(s_2-s_1)/(n_kernels+2)];
for i = 1 : size(f,1)
f(i,:) = exp(-((s-f_peaks(i))/var_basis).^2);
%Conversion to mass concentration
f(i,:) = f(i,:).*s_exp.^3*1/6*pi();
end
end

f_norm = max(f');
f = (f'./repmat(max(f'),size(f,2),1))';

%Kernels normalized to one everywhere.
F=sum(f);
for i = 1 : size(f,1)
f(i,:) = f(i,:)./F;  
end
f_0 = f;

%Plot kernels
figure(4); h = semilogx(s_exp,f'); title('Kernel functions normalized to one');
set(h,'linewidth',2);
set(gca,'xlim',[exp(s_1) exp(s_2)]);
set(gca,'ylim',[0 1.05]);
set(gca,'fontsize',14);
set(gca,'linewidth',1);
set(4,'renderer','painters');
print(4,'-r200','-dpng','normalized_kernels_1.png');
figure(5); h = semilogx(s_exp,f'); title('Modified kernel functions summing up to one everywhere');
set(h,'linewidth',2);
set(gca,'xlim',[exp(s_1) exp(s_2)]);
set(gca,'ylim',[0 1.05]);
set(gca,'fontsize',14);
set(gca,'linewidth',1);
set(5,'renderer','painters');
print(5,'-r200','-dpng','normalized_kernels_3.png')

%Particle mass concentration
g=zeros(n_test,length(s));
sigma_val = 1.23; 
C_scale = 5.11E4;
peak_val = [0.008 0.02 0.1 0.3 0.8 9];
for i = 1 : length(peak_val)
g(i,:)=C_scale./(sqrt(2*pi())*log(sigma_val)).*exp(-0.5*((s-log(peak_val(i))).^2)./log(sigma_val).^2);
g(i,:)=1/6*pi()*g(i,:).*s_exp.^3; 
end


%L1-inversion, kernel basis
A_ker = zeros(size(f,1), size(f,1));
for i = 1 : size(f,1)
for j = 1 : size(f,1)
A_ker(i,j) = sum((s(2:end)-s(1:end-1)).*(f_0(i,2:end).*f_0(j,2:end)));
end
end

B = zeros(size(f,1), size(f,1));
for i = 1 : size(f,1)
for j = 1 : size(f,1)
B(i,j) = sum((s(2:end)-s(1:end-1)).*f(i,2:end).*f(j,2:end));
end
end

for n_iterations = 1 : size(f,1)
C{n_iterations} = zeros(size(f,1), size(f,1));
for i = 1 : size(f,1)
for j = 1 : size(f,1)
C{n_iterations}(i,j) = sum((s(2:end)-s(1:end-1)).*f(n_iterations,2:end).*f(i,2:end).*f(j,2:end));
end
end
F_mat{n_iterations} = B\C{n_iterations};
end

%L1-inversion
A_pnt = zeros(size(f,1), size(s,2));
for i = 1 : size(f,1)
A_pnt(i,1) = 0; 
for j = 2 : size(s,2)
A_pnt(i,j) = (s(j)-s(j-1)).*f_0(i,j);
end
end

A_pnt_aux = A_pnt'*A_pnt;

B_2 = diag([1 (s(2:end)-s(1:end-1))]);

for k = 1 : size(f,1)
C_2{k} = zeros(size(s,2), size(s,2));
C_2{k}(1,1) = 0;
for i = 2 : size(s,2)
j = i;
C_2{k}(i,j) = (s(j)-s(j-1)).*f(k,j);
end
F_mat_2{k} = diag(B_2\C_2{k});
end

%Run the test cases

for n=1:n_test
    
    data=[];

%Simulated data
y = zeros(size(f,1),1); 
for i = 1 : size(y,1)
    y(i) = sum((s(2:end)-s(1:end-1)).*g(n,2:end).*(f_0(i,2:end)));
end

y_aux = y; 

y_data = zeros(sample_size,5,4); 
for ell = 1 : length(noise)

for iter_ind_aux = 1 : sample_size

%IAS iteration, kernel-based
y = y_aux + max(y_aux).*noise(ell).*randn(size(y_aux));
tic;

x = zeros(12,1);
for l = 1 : size(f,1)
x = x + A_ker(l,:)'*y(l); 
end 

theta = sqrt((x).^2);

for j = 1 : n_iterations
    
    x = (A_ker'*A_ker + L1_reg_param*((diag(1./theta))))\(A_ker'*y);
            theta = sqrt((x).^2) + L1_eps_val;
 
end
time_val = toc;
z = zeros(1,size(s,2));
for i = 1 : size(f,1)
    z = z + x(i)*f_0(i,:);
end

L1_ker=z;
time_L1_ker(ell,iter_ind_aux,n) = time_val;
norm_L1_ker_1(ell, iter_ind_aux, n) = sum((s(2:end)-s(1:end-1)).*abs(z(2:end) - g(n,2:end)))./sum((s(2:end)-s(1:end-1)).*abs(g(n,2:end)));
norm_L1_ker_inf(ell, iter_ind_aux, n) = max(abs(z - g(n,:)))./max(abs(g(n,:)));


tic;

%IAS iteration, point-wise

x = zeros(size(s,2),1);
for l = 1 : size(f,1)
x = x + f_0(l,:)'*y(l); 
end 

theta = sqrt((x).^2) + L1_eps_val;

for j = 1 : n_iterations
    x = (A_pnt_aux + L1_reg_param*((diag(1./theta))))\(A_pnt'*y);
            theta = sqrt((x).^2) + L1_eps_val;
end
time_val = toc;
z = x';
L1_pnt=z;
time_L1_pnt(ell,iter_ind_aux,n) = time_val;
norm_L1_pnt_1(ell, iter_ind_aux, n) = sum((s(2:end)-s(1:end-1)).*abs(z(2:end) - g(n,2:end)))./sum((s(2:end)-s(1:end-1)).*abs(g(n,2:end)));
norm_L1_pnt_inf(ell, iter_ind_aux, n) = max(abs(z - g(n,:)))./max(abs(g(n,:)));

if n == 1
    value=max(abs(z-g(n,:)))./max(abs(g(n,:)));
end


%Fixed-point inversion

tic;
x = zeros(size(s,2),1);
for l = 1 : size(f,1)
x = x + f_0(l,:)'*y(l); 
end 

for l=1:n_iterations


%Fixed-point, point-wise
x_aux = zeros(size(x));
for i = 1 : size(f,1)
x_aux = x_aux + y(i)*F_mat_2{i}.*x./(A_pnt(i,:)*x);
end
x = x_aux;
end

time_val = toc;
z = x';
fixed_point_pnt = z;
time_fp_pnt(ell,iter_ind_aux,n) = time_val;
norm_fp_pnt_1(ell,iter_ind_aux,n) = sum((s(2:end)-s(1:end-1)).*abs(z(2:end) - g(n,2:end)))./sum((s(2:end)-s(1:end-1)).*abs(g(n,2:end)));
norm_fp_pnt_inf(ell,iter_ind_aux,n) = max(abs(z - g(n,:)))./max(abs(g(n,:)));

tic
x = zeros(12,1);
for l = 1 : size(f,1)
x = x + A_ker(l,:)'*y(l); 
end 

for l=1:n_iterations

%Fixed-point, kernel-based
x_aux = zeros(size(x));
for i = 1 : size(f,1)
x_aux = x_aux + y(i)*F_mat{i}*x./(A_ker(i,:)*x);
end
x = x_aux;

end

time_val = toc;
z = zeros(size(z));
for i = 1 : size(f,1)
    z = z + x(i)*f_0(i,:);
end

fixed_point_ker = z;
time_fp_ker(ell,iter_ind_aux,n) = time_val;
norm_fp_ker_1(ell,iter_ind_aux,n) = sum((s(2:end)-s(1:end-1)).*abs(z(2:end) - g(n,2:end)))./sum((s(2:end)-s(1:end-1)).*abs(g(n,2:end)));
norm_fp_ker_inf(ell,iter_ind_aux,n) = max(abs(z - g(n,:)))./max(abs(g(n,:)));

end 

%Plot reconstructed distributions
if ell == 2
figure(1); clf; set(gcf,'renderer','painters');
semilogx(s_exp, g(n,:),'-','color',0.75*[1 1 1], 'linewidth',7);
hold on
semilogx(s_exp, fixed_point_ker,'-','color',0.9*[0 1 1], 'linewidth',2);
semilogx(s_exp, fixed_point_pnt,'-.','color',0.7*[0 1 0], 'linewidth',4);
semilogx(s_exp, L1_ker,'-.','color',0.8*[1 0 1], 'linewidth',2);
semilogx(s_exp, L1_pnt,':','color',0*[1 1 1], 'linewidth',2);
legend('Original','FP_{ker}','FP_{pnt}','L1_{ker}','L1_{pnt}','location','northeastoutside','orientation','vertical')
xlabel('Particle size (µm)')
ylabel('Particle concentration (µg/m^3)')
title(['Simulated distribution ' int2str(n)])
set(gca,'xlim',[0 10])
set(gca,'ylim', [0 1.5*max([max(g(n,:))])]);
set(gca,'linewidth',0.5,'fontsize',12)
set(gca,'fontsize',14);
drawnow
print(1,'-r300','-dpng',['simuloitu_jakauma_2per_' int2str(n) '.png'])
end 
y_data_1(:,ell,1) = norm_fp_ker_1(ell,:,n)';
y_data_1(:,ell,2) = norm_fp_pnt_1(ell,:,n)';
y_data_1(:,ell,3) = norm_L1_ker_1(ell,:,n)';
y_data_1(:,ell,4) = norm_L1_pnt_1(ell,:,n)';
y_data_2(:,ell,1) = norm_fp_ker_inf(ell,:,n)';
y_data_2(:,ell,2) = norm_fp_pnt_inf(ell,:,n)';
y_data_2(:,ell,3) = norm_L1_ker_inf(ell,:,n)';
y_data_2(:,ell,4) = norm_L1_pnt_inf(ell,:,n)';

end

      y_data = y_data_1(:);
      legenda=[repmat('FP_{ker}',5*sample_size,1);repmat('FP_{pnt}',5*sample_size,1);repmat('L1_{ker}',5*sample_size,1);repmat('L1_{pnt}',5*sample_size,1)];
      kohina=[repmat('1.0%',sample_size,1);repmat('1.5%',sample_size,1);repmat('2.0%',sample_size,1);repmat('2.5%',sample_size,1);repmat('3.0%',sample_size,1)];
      kohina = repmat(kohina,4,1);
      [y_data,x_data,indeksi] = tab2box(kohina,y_data,legenda);
      
      % sort
      IX = [1 2 3 4]; % order
      indeksi = indeksi{1}(IX);
      y_data = y_data(:,:,IX);
      I_aux = find(isnan(y_data));
      
      
      %Plot histograms
      figure(2); clf;
      h = boxPlot(x_data,100*y_data,...
          'boxColor',{0.9*[0 1 1],0.7*[0 1 0],0.8*[1 0 1],1*[1 1 1]},'medianColor','k',...
          'scalewidth',true,'xseparator',true,'notch',true,'notchdepth',0.3,...
          'groupLabels',indeksi,'showOutliers',false,'limit',[10 90]);
      h_legend = get(gca,'legend');
      set(h_legend,'location','northeastoutside');
      box on
      ylim('auto')
      ytickformat('percentage')
      xlabel('Noise percentage')
      ylabel('Relative error')
      set(gca,'fontsize',14);
      title(['Simulated distribution ' int2str(n)]);
      print(2,'-r300','-dpng',['1_boxplot_jakauma_' int2str(n) '.png'])
      
      y_data = y_data_2(:);
      legenda=[repmat('FP_{ker}',5*sample_size,1);repmat('FP_{pnt}',5*sample_size,1);repmat('L1_{ker}',5*sample_size,1);repmat('L1_{pnt}',5*sample_size,1)];
      kohina=[repmat('1.0%',sample_size,1);repmat('1.5%',sample_size,1);repmat('2.0%',sample_size,1);repmat('2.5%',sample_size,1);repmat('3.0%',sample_size,1)];
      kohina = repmat(kohina,4,1);
      [y_data,x_data,indeksi] = tab2box(kohina,y_data,legenda);
      
      % sort
      IX = [1 2 3 4]; % order
      indeksi = indeksi{1}(IX);
      y_data = y_data(:,:,IX);
      I_aux = find(isnan(y_data));
      
      
      %Plot histograms
      figure(3); clf;
      h = boxPlot(x_data,100*y_data,...
          'boxColor',{0.9*[0 1 1],0.7*[0 1 0],0.8*[1 0 1],1*[1 1 1]},'medianColor','k',...
          'scalewidth',true,'xseparator',true,'notch',true,'notchdepth',0.3,...
          'groupLabels',indeksi,'showOutliers',false,'limit',[10 90]);
      h_legend = get(gca,'legend');
      set(h_legend,'location','northeastoutside');
      box on
      ylim('auto')
      ytickformat('percentage')
      xlabel('Noise percentage')
      ylabel('Relative error')
      title(['Simulated distribution ' int2str(n)]);
      set(gca,'fontsize',14)
      print(3,'-r300','-dpng',['inf_boxplot_jakauma_' int2str(n) '.png'])

end

time_fp_ker = mean(time_fp_ker(:))
time_fp_pnt = mean(time_fp_pnt(:))
time_L1_ker = mean(time_L1_ker(:))
time_L1_pnt = mean(time_L1_pnt(:))
