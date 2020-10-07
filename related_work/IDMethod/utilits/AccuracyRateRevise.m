function [DR,TP,FP,TN,FN] = AccuracyRate(2)(GI,DI,N1,N2,path,deN)
% AccuracyRate provides the detection success rate after the comparison of
% the Ground-truth image and Detected image by a specific method
% 
% [OUTPUT]
% DR : Detection Result in DSR, Sensitivity, Specificity, PPV and NPV
% TP  : True Positive
% FP  : False Positive
% TN  : True Negative
% FN  : False Negative
%
% [INPUT]
% GI  : Ground-truth image
% DI  : Detected image
% N1  : row's size of a repetitive unit in the BB/RB method
% N2: : column's size of a repetitive unit in the BB/RB method
% path: the path for saving the TP,FP,TN,FN's images
% deN : defect name
%
% Written by Henry Y.T. Ngan, Dept. of Mathematics,
% Hong Kong Baptist University
%
% Created on 10 May, 2013
% Updated on 8 Aug, 2013

% C-input ground true, A - resized input GI

% N1=uint8(N1);
% N2=uint8(N2);

% GI is in [0,255]
% re-assign the pixel values of the input ground-truth image in the range
% of [0,1]
nGI=zeros(256,256);
for i=1:256,
    for j=1:256,
        if GI(i,j)==255
           nGI(i,j)=1;
        else nGI(i,j)=0;       
        end 
    end
end

% Resize the input ground-truth image (re-valued) as the detected image
rGI=nGI(N2:256,N1:256);

% true positive
TPi=zeros(256-N2+1,256-N1+1);
for i=1:256-N2+1,
    for j=1:256-N1+1,
        if (rGI(i,j)==1) && (DI(i,j)==1)
           TPi(i,j)=1;
        else TPi(i,j)=0;
        end
    end
end
figure(1);imshow(TPi); 

% false positive
FPi=zeros(256-N2+1,256-N1+1);
for i=1:256-N2+1,
    for j=1:256-N1+1,
        if (rGI(i,j)==0) && (DI(i,j)==1)
           FPi(i,j)=1;
        else FPi(i,j)=0;
        end
    end
end
figure(2);imshow(FPi); 

% true negative
TNi=zeros(256-N2+1,256-N1+1);
for i=1:256-N2+1,
    for j=1:256-N1+1,
        if (rGI(i,j)==0) && (DI(i,j)==0)
           TNi(i,j)=1;
        else TNi(i,j)=0;
        end
    end
end
figure(3);imshow(TNi);

% false negative
FNi=zeros(256-N2+1,256-N1+1);
for i=1:256-N2+1,
    for j=1:256-N1+1,
        if (rGI(i,j)==1) && (DI(i,j)==0)
           FNi(i,j)=1;
        else FNi(i,j)=0;
        end
    end
end
figure(4);imshow(FNi);

% check the number of pixles whether is equal to the size of detected image
% T=[sum(sum(TPi)) sum(sum(FPi)) sum(sum(TNi)) sum(sum(FNi))]  
 
TP=sum(sum(TPi));
FP=sum(sum(FPi));
TN=sum(sum(TNi));
FN=sum(sum(FNi));

DSR= (TP+TN)/(TP+FP+TN+FN)*100; % Detection success rate
Sen = TP/(TP+FN)*100; % Sensitivity 
Spec = TN/(FP+TN)*100; % Specificity
PPV = TP/(TP+FP)*100; % PPV
NPV = TN/(TN+FN)*100; % NPV

defect=sprintf(deN)
DR.TP=TP;
DR.FP=FP;
DR.TN=TN;
DR.FN=FN;
DR.DSR= DSR; % Detection success rate
DR.Sen = Sen; % Sensitivity 
DR.Spec = Spec; % Specificity
DR.PPV = PPV; % PPV
DR.NPV = NPV % NPV

% Save figure and values
figure(1) % TPi
rootname=deN;
rootname1='_TP';
ext='.bmp';
file_name = [rootname,rootname1];
saveas(gcf, [path, file_name, ext], 'bmp');

figure(2) % FPi
rootname2='_FP';
ext='.bmp';
file_name = [rootname,rootname2];
saveas(gcf, [path, file_name, ext], 'bmp');

figure(3) % TNi
rootname3='_TN';
ext='.bmp';
file_name = [rootname,rootname3];
saveas(gcf, [path, file_name, ext], 'bmp');

figure(4) % FNi
rootname4='_FN';
ext='.bmp';
file_name = [rootname,rootname4];
saveas(gcf, [path, file_name, ext], 'bmp');

ext='.mat';
savefile = [path,rootname];
save(savefile,'DR');