%%%%% function for testing stage
% InputImage: the defective test sample
% win: winning therhold value of score
% lose: losing threhsold value of score
% light: light threshold value of ELO point
% darkL dark threshold value of ELO point
% ResultantImage: the resultant image
% ELO: the ELO point matrix of the defective test sample

function [ResultantImage,ELO]=TestingStage(InputImage,win,lose,light,dark)
global xp yp xsize ysize RandomPartitionX RandomPartitionY

   % Pre-processing
   X=PreProcessing(InputImage);
   
   
% Compute the ELO matrix of the input image      
  ELO=zeros(xsize,ysize);
   for n=1:xsize
      for m=1:ysize
          ELO(n,m)=1000;
      end
   end
    for n=1:xsize
        for m=1:ysize
          tempA=X(n:n+xp-1,m:m+yp-1);
             for p=1:RandomPartitionX
                 xSchedule=randperm(xsize);
                 a=xSchedule(p);
                  for q=1:RandomPartitionY
                     ySchedule=randperm(ysize);
                     b=ySchedule(q);
                     tempB=X(a:a+xp-1,b:b+yp-1); 
                     ELO(n,m)=EloUpdate(ELO(n,m),ELO(p,q),tempA,tempB,win,lose);
                  end
             end
        end
    end

% Determine which partitions is defect 
    U=zeros(xsize,ysize);


    for n=1:xsize
        for m=1:ysize
            if ELO(n,m)>light
                U(n,m)=1;
            else if ELO(n,m)<dark
                    U(n,m)=0.5;
                else U(n,m)=0;
                end
            end
        end
    end
              
% Apply filter to produce the final resultant image  
ResultantImage=medfilt2(U);

  
end

% Copyright (c) 2016, Colin S.C. Tsang
% All rights reserved. Draft date: 1st September 2014. Release date: 1st April 2017.
% 
% Notifications:
% 1. The code is based on the following publication: 
%    Colin SC Tsang, Henry YT Ngan, and Grantham KH Pang, "Fabric inspection based on the Elo rating method.", Pattern Recognition 51 (2016):378-394 
% 2. Please cite the above published paper as reference if you use the codes for any purpose.
% 3. Code Author: Colin SC Tsang (Email: colintsang@life.hkbu.edu.hk)
% 4. Hong Kong Baptist University legally has the intellectual properties of the codes. 
% 
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 1. The code should only be used for academic purpose or research, not for any business or commercial activities.
% 2. Redistributions of source code must retain the above copyright, the above notifications, this list of conditions, the following database regulations, and the following disclaimer.
% 3. Redistributions in binary form must reproduce the above copyright, the above notifications, this list of conditions, the following database regulations, the following disclaimer in the documentation and/or other materials provided with the distribution.
% 
% Database regulations:
% Users should agree the following regulations before they use the database for any purpose:
% 1. The database* should only be used for academic purpose or research, not for any business or commercial activities.
% 2. The database can be used by yourself only. Please do not expose the database to others without our priori approval.
% 3. Please mention the source of this database ( i.e. Industrial Automation Research Laboratory, Dept. of Electrical and Electronic Engineering, The University of Hong Kong) in any publication , such as conference proceeding, journals, etc, related to the images of this database.
% *The database here means the fabric images.
% 
% Disclaimer:
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
% AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
% IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
% ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
% LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
% CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
% INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
% CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
% POSSIBILITY OF SUCH DAMAGE.
