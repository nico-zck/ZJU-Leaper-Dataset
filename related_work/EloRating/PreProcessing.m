%%%%% Function for pre-processing
% OriImage: the input image sample
% ResultImage: lv2 harr wavelet transformed apprximated image

function ResultImage=PreProcessing(OriImage)

   [n,n,k]=size(OriImage);
   if size(OriImage, 3) == 3
       A=rgb2gray(OriImage);
   else
       A=OriImage;
   end
   B=im2double(A);
 
  [C,S]=wavedec2(B,2,'haar');
   D=wrcoef2('a',C,S,'haar',2);

   
   m=n/4;
        ResultImage=zeros(m);

       for   i=1:m
           for   j=1:m
               ResultImage(i,j)=D(i*4,j*4);
          end
       end
 

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
