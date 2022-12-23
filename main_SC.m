clear; close all; clc;
%% Load and preprocess test data
addpath('load_imagetool');
DCDL_multiBlock_city = load('filters_BPEGM_CDL_multiBlock_obj9.73e+03_city.mat');
method = {'CDL_multiBlock_city'};
params.method = method;
test_image=mygetdirfiles('test_images');
test_imagecell=load_images(test_image);
index_a = 1:2:size(test_imagecell);
index_b = 2:2:size(test_imagecell);
A_source=cell(size(index_a,2),1);
B_source=cell(size(index_b,2),1);
k = 1;%%显示方法
t = 1;%%控制方法行数
q = 1;
m = 1;%%sheet
%% Parameters
flag = 2;
param.size_kernel = [11, 11, 100];    %the size and number of 2D filters
param.alpha = 1;              %reg. param.: alpha in DOI: 10.1109/TIP.2017.2761545
param.major_type = {'D', 'D'};        
%Options for BPEG-M algorithms
param.arcdegree = 95;    %param. for gradient-based restarting: angle between two vectors (degree)
param.max_it = 500;       %max number of iterations
param.tol = 1e-4;        %tol. val. for the relative difference stopping criterion
%Fixed random initialization
s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
init.z = [];
verbose_disp = 1;    %option: 1, 0
saving = 1;   %option: 1, 0
for i = 1:numel(test_imagecell)/2
    f = test_image{index_a(i)};
    [p, n, x] = fileparts(f);
    params.p = p;
    params.n = n;
    params.x1 = x;
    A_source{i} = test_imagecell{index_a(i)};
    B_source{i} = test_imagecell{index_b(i)};     
     %%CDL_multiBlock_city
    init.d = DCDL_multiBlock_city.d;  
y_F_CDL_multiBlock_city{i}=CDL_multiBlock_Fusion(A_source{i},B_source{i},param,verbose_disp, init,flag);
    t = t + size(method,2) + 1;
    q = q + 3;
    k = 1;
%%%将各种方法所得到的融合图像放到同一个大的矩阵中
   result = cat(3,y_F_CDL_multiBlock_city{i);
   conf.fusion_image{i} = {};
%%%%将融合图像写入到指定的文件夹里
   for j = 1:numel(method)
        conf.fusion_image{i}{j} = fullfile(p, 'results', [n sprintf('[%d-%s]', j, method{j}) x]);
        imwrite(uint8(result(:, :, j)), conf.fusion_image{i}{j});%%%%将各种方法的结果放到result文件夹中
    end
end


