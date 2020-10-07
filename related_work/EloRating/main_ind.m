clc;
close all;
clear;

% Basic setting
global xp yp xsize ysize RandomPartitionX RandomPartitionY constantK wvariable
iptsetpref('ImshowBorder','tight');
iptsetpref('ImshowInitialMagnification','fit');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Change parameters here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xp and yp are length and width of partitions.
% xsize and ysize is the length and width of pre-processed image that can
% be slided by the golden image.
% RandomPartitionX is the number of randomly selected rows.
% RandomPartitionY is the number of randomly selected columns.
% wvariable is w-variable in Elo's formula.
% constantK is constant K in Elo's formula.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% According to our publication,
% the optimised sets of parameters {xp,yp,RandomPartitionX*RandomParitionY,wvariable,constantK}
% are {7,4,15,400,16}, {7,4,40,100,10}, and {10,6,15,400,16} for
% dot-,star-,box-patterned fabrics, respectively.
% *one may skip median filter in the testing stage for box-patterned fabrics
% to get better result, as recommened in the paper.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

xp=8;
yp=8;
xsize=64-xp+1;
ysize=64-yp+1;
RandomPartitionX=4;
RandomPartitionY=4;
wvariable=400;
constantK=8;

p_id_list = [5];
for p_id=p_id_list
    p_str = ['I',int2str(p_id)];
    fprintf(['=======', p_str, '=======\n']);
    
    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Traning stage start
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    train_names = textread(['D:/Dataset/Dataset-Industry/Industry-Final/ImageSets/Patterns/p',int2str(p_id),'_train_unsupervised.txt'], '%s');
    num_train = length(train_names);

    % Input defect-free images
    for i=1:num_train
        image = imread(['D:/Dataset/Dataset-Industry/Industry-Final/Images/', train_names{i}, '.jpg']);
        R{i} = imresize(image, [256 256]);
    end

    % lv2 harr wavelet is applied to each images
    for k=1:num_train
        A{k}=PreProcessing(R{k});
    end


    % Extract golden image
    GI=zeros(xp,yp,num_train);
    score=zeros(xsize,ysize,num_train);
    for k=1:num_train
        GI(1:xp,1:yp,k)=A{k}(1:xp,1:yp);
    end


    % Compute the score of each game while the golden image sliding
    for k=1:num_train
        for n=1:xsize
            for m=1:ysize
                tempGI=A{k}(n:n+xp-1,m:m+yp-1);
                score(n,m,k)=mean2(GI(1:xp,1:yp,k)-tempGI);
            end
        end
    end


    % Compute the win threshold value and lose threshold value
    FinalWinThreshold=max(max(max(score(1:xsize,1:ysize,1:num_train))));
    FinalLoseThreshold=min(min(min(score(1:xsize,1:ysize,1:num_train))));

    % Compute ELO matrix of the 5 defect free samples
    ELO=zeros(xsize,ysize,num_train);
    for k=1:num_train
        for n=1:xsize
            for m=1:ysize
                ELO(n,m,k)=1000;
            end
        end
    end

    for k=1:num_train
        for n=1:xsize
            for m=1:ysize
                tempA=A{k}(n:n+xp-1,m:m+yp-1);
                for p=1:RandomPartitionX
                    xSchedule=randperm(xsize);
                    a=xSchedule(p);
                    for q=1:RandomPartitionY
                        ySchedule=randperm(ysize);
                        b=xSchedule(q);
                        tempB=A{k}(a:a+xp-1,b:b+yp-1);
                        [ELO(n,m,k),ELO(a,b,k)]=EloUpdate(ELO(n,m,k),ELO(a,b,k),tempA,tempB,FinalWinThreshold,FinalLoseThreshold);
                    end
                end
            end
        end
    end



    % Compute the light threshold value and dark threshold value
    LightThreshold=zeros(num_train,1);
    DarkThreshold=zeros(num_train,1);

    for k=1:num_train
        LightThreshold(k)=max(max(ELO(1:xsize,1:ysize,k)));
        DarkThreshold(k)=min(min(ELO(1:xsize,1:ysize,k)));
    end

    FinalLightThreshold=(sum(LightThreshold))/num_train;
    FinalDarkThreshold=(sum(DarkThreshold))/num_train;

    fprintf([p_str, ' training is done.\n']);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Traning stage end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Testing stage start
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	test_names = textread(['D:/Dataset/Dataset-Industry/Industry-Final/ImageSets/Patterns/p',int2str(p_id),'_test.txt'], '%s');
    num_test = length(test_names);

    % Input defect-free images
    for i=1:num_test
        image = imread(['D:/Dataset/Dataset-Industry/Industry-Final/Images/', test_names{i}, '.jpg']);
        B{i} = imresize(image, [256 256]);
    end

    % Initialization
    for k=1:num_test
        defective{k}=zeros(xsize,ysize);
    end
    dfELO=zeros(xsize,ysize,num_test);

    % Apply testing stage to output the resultant images
    for k=1:num_test
        [defective{k},dfELO(:,:,k)]=TestingStage(B{k},FinalWinThreshold,FinalLoseThreshold,FinalLightThreshold,FinalDarkThreshold);
    end
    
    fprintf([p_str, ' test is done.\n']);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Testing stage end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %% Show the resultant images of hole-defect
%     figure;
%     for i=1:9
%         subplot(3,3,i)
%         imshow(B{i})
%     end
%     figure;
%     for i=1:9
%         subplot(3,3,i)
%         imshow(defective{i})
%     end
    % for i=1:num_test
    %     imshow(defective{i})
    %     pause;
    % end

    %% save
    [h, w, c] = size(R{1});
    results = zeros(num_test, h, w);
    for i=1:num_test
        results(i,:,:) = imresize(defective{i}, [h,w], 'nearest');
    end
    names = test_names;
    save(['ER_', p_str ,'.mat'], 'results', 'names', '-v7.3');
end
