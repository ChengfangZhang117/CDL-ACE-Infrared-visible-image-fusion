function [imgf] = CDL_multiBlock_Fusion( x1, x2,param, verbose_disp, init,flag )
    if size(x1,3)>1
        x1_gray=double(rgb2gray(x1));
        x2_gray=double(rgb2gray(x2));
    else
       x1_gray = x1;
       x2_gray = x2;
     end
    npd = 16;
    fltlmbd = 5;
    [s1_l, s1_h] = lowpass(double(x1_gray), fltlmbd, npd);
    [s2_l, s2_h] = lowpass(double(x2_gray), fltlmbd, npd);
    %% 求解稀疏系数via BPEG-M
    disp('BPEG-M for first source test image');
    [z1, Dz1,psf_radius1,psf_s1] = BPEGM_SC_2D_multiBlk( s1_h, param.size_kernel, ...
    param.alpha, param.arcdegree, param.major_type(1), param.major_type(2), ...
    param.max_it, param.tol, verbose_disp, init);
    disp('BPEG-M for two source test image');
    [z2, Dz2,psf_radius2,psf_s2] = BPEGM_SC_2D_multiBlk(s2_h, param.size_kernel, ...
    param.alpha, param.arcdegree, param.major_type(1), param.major_type(2), ...
    param.max_it, param.tol, verbose_disp, init);
    A1=sum(abs(z1),4);
    A2=sum(abs(z2),4);
    
%     plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],obj_val1,'k-',[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],z_relErr1,'b--');%对比数据集合
%  title('误差对比图');%标题
%  xlabel('邻居数目K');
%  ylabel('mae');
%  legend('obj','稀疏系数');
%  %制作图例
%  grid on;
%  
%  plot([10,20,30],obj_val1,'k-');%对比数据集合
%  title('误差对比图');%标题
%  xlabel('邻居数目K');
%  ylabel('mae');
%  legend('obj');
%  %制作图例
%  grid on;
%%   
if flag == 1  
    r=9;  
else
    r=3; 
end

ker=ones(2*r+1,2*r+1)/((2*r+1)*(2*r+1));
AA1=imfilter(A1,ker);
AA2=imfilter(A2,ker);
decisionMap=AA1>AA2;

%base layer fusion
if flag == 1  
    imgf_l=s1_l.*decisionMap((1 + psf_radius1):(size(decisionMap,1) - psf_radius1),(1 + psf_radius1):(size(decisionMap,2) - psf_radius1))+s2_l.*(1-decisionMap(1 + psf_radius1:size(decisionMap,1) - psf_radius1,1 + psf_radius1:size(decisionMap,2) - psf_radius1));
else
    imgf_l=(s1_l+s2_l)/2;
end

%detail layer fusion
[height,width]=size(A1);
X=z1;
for j=1:height
    for i=1:width
        for k = 1:100
        if decisionMap(j,i)==0
            X(j,i,:,k)=z2(j,i,:,k);
        end
        end
    end
end
dnorm = @(x) bsxfun(@rdivide, x, sqrt(sum(sum(x.^2, 1), 2)));
d = dnorm(init.d);
PSt = @(u) d_to_dpad(u, [size(s1_h,1) + psf_s1 - 1 size(s1_h,2) + psf_s1 - 1 1], [psf_s1  psf_s1 100], [psf_radius1 + 1  psf_radius1 + 1]);
d_hat = fft2( PSt(d) );
z_hat = reshape( fft2( reshape(X, size(s1_h,1) + psf_s1 - 1,size(s1_h,2) + psf_s1 - 1,[]) ), [ size(s1_h,1) + psf_s1 - 1   size(s1_h,2) + psf_s1 - 1   1     100] );
imgf_h1 = real(ifft2( reshape(sum(repmat(d_hat,[1,1,1,1]) .* permute(z_hat,[1,2,4,3]), 3), [size(s1_h,1) + psf_s1 - 1 size(s1_h,2) + psf_s1 - 1 1]) ));
imgf_h2 = imgf_h1(1 + psf_radius1:end - psf_radius1,1 + psf_radius1:end - psf_radius1);
%% fusion
disp('fusion');
if size(x1,3)>1
    decisionMap = guidedfilter(imgf_h2,decisionMap((1 + psf_radius1):(size(decisionMap,1) - psf_radius1),(1 + psf_radius1):(size(decisionMap,2) - psf_radius1)),8,0.1);
    decisionMap_rgb=repmat(decisionMap,[1 1 3]);
    imgf=(double(x1)/255).*decisionMap_rgb+(double(x2)/255).*(1-decisionMap_rgb);
    imgf=uint8(imgf);
else 
    imgf=uint8(imgf_l+imgf_h2);
end


