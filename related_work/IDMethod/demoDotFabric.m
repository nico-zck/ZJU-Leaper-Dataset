%%% This code handle the defect detection for Dot-patterned fabric
%%% problem by a simple image decomposion rationale.
%%%% Dot package: 
%%%%     'b' --BrokenEnd;          'h' --Hole;         'k' --knots;
%%%%     'n' --NettingMultiple;    'tt'--ThickBar;     't' --ThinBar;  
%%%%     'L' --LoosePick;          'm' --MissPick;     'op'--OilWarp; 
%%%%     'ot'--OilWeft
%%%% written by Wenxing Zhang, wenxing84@gmail.com 2012
clc; 
close; 
clear;

%%%%%%%%%%%%%%%%%%% Test images %%%%%%%%%%%%%%%%%%%%%%%%%%%%
types = {'b','h','k','n','tt','t'};
idx = 1;
for t=1:length(types)
    d_type = types{t};   %type of defect
    for i = 1:5  %number of image
        name = ['bad_', d_type, int2str(i), '.png'];
        IM = ['../dataset/dot/test/', name];  
        I  = im2double(imread(IM)); x0 = rgb2gray(I); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%%%%% Dot pattern
        switch d_type
            case {'b','h'}
                tau=0.3; mu=1; beta1=50; beta2=1;  %BrokenEnd,Hole 
            case 'k'
                tau=0.5; mu=1; beta1=100; beta2=1; %Knot
            otherwise 
                tau=0.2; mu=1; beta1=80; beta2=1;  %other defects
        end

        %%%% solver
        opts.MaxIt = 100; p =inf;   
        opts.tau = tau; opts.mu = mu; opts.beta1=beta1; opts.beta2=beta2;
        [u_S,v_S] = ADM(x0,p,opts);
        %%%%%%% display defects
        
        results(idx,:,:) = u_S;
        names(idx) = string(name);
        idx = idx + 1;
        
%         figure;
%         subplot(121); imshow(I,[]); title('original')
%         subplot(122); imshow(u_S,[]);title('detected defect'); 
    end
end

save('ID_dot.mat', 'results', 'names');
