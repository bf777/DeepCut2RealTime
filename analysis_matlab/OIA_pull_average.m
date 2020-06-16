function [SM,SMs,ST] = OIA_pull_average(I,J,SF,refper,c,r,win,winA)
% function [SM,SMs,ST] = OIA_pull_average(I,J,SF,refper,c,r,win,winA)
% input:
% I = [X,Y,F] imaging
% J = timestamp (s)
% SF = sampling freq (Hz)
% win = temporal window (how many sec before and after trig), eg: [-5 5]
% winA = window of analysis: 0= full sequence [Tmin Tmax] = time position (sec)
%                          e.g. [0 600] = first 600s
% refper = refractory period [min max] (sec), eg: [3 4]
%          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
%          WARNING: if negative, apply both way !!!!!!!
%          !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
% c = global activity substraction (if 1)
% r = resize (x), if negative, random pulls are really random...
% output:
% SM = [X,Y,F,T] of pull
% SMs = idem but no pull (random)
% ST = no DF/F

real_random = 0; if r < 0, r = -r; real_random = 1; end;

if refper(2)<0
    refper = abs(refper);
    tw = 1;
else
    tw = 0;
end

if length(winA) == 2
    Jc = J((J>winA(1))&(J<winA(2)));
else 
    Jc = J;
end

Jc = Jc(Jc>0); % no zero and no <0
good = [];
if tw == 0
    for i = 2:length(Jc)
        d = Jc(i)-Jc(i-1);
        if ((d>refper(1))&(d<refper(2)))
            good = [good i];
        end
    end
else
    for i = 2:length(Jc)-1
        d = Jc(i)-Jc(i-1);
        d2 = Jc(i+1)-Jc(i);
        if (((d>refper(1))&(d<refper(2)))&((d2>refper(1))&(d2<refper(2))))
            good = [good i];
        end
    end
end
Jc = Jc(good);

%Js = -win(1) + rand(size(Jc)) .* max(Jc)-(win(2)-win(1));
%Js = sort(Js);

if real_random == 0, limite = max(abs(win)); else, limite=0; end
win = round(win.*SF);
winsize = win(2)-win(1)+1;

SM = zeros(round(size(I,1)*r),round(size(I,2)*r),winsize,length(Jc)-2,'single'); 
SMs = zeros(round(size(I,1)*r),round(size(I,2)*r),winsize,length(Jc)-2,'single'); 
ST = zeros(size(I,1),size(I,2),winsize,length(Jc)-2,'uint16');
for i = 2:length(Jc)-1
    loca = round(Jc(i)*SF);
    disp(['pull ' num2str(i) '/' num2str(length(Jc)-1) ' (t=' num2str(Jc(i)) 's)'])
 
    try
        S = single(I(:,:,loca+win(1):loca+win(2)));
        
        ST(:,:,:,i) = uint16(S);
        S = imresize(S,r,'nearest');       
        if c == 1, v.method = 1; v.dc = 1; S = OIA_SA_globalactivity(S,v); end
        if win(1)<0
            M = mean(S(:,:,1:abs(win(1))),3);
        else
            M = mean(S(:,:,1:size(S,3)),3);
        end
        for f = 1:size(S,3)
            S(:,:,f) = 100*(S(:,:,f) - M)./M;
        end
        SM(:,:,:,i) = S;  
    catch
        disp(' > out of range')
    end
end

n_blank = 0;
while n_blank < (size(SM,4)-1)
    
    position =  -win(1) + rand(1) .* max(Jc)-(win(2)-win(1)) ; % generate a randomw position (s)
    loca = round(position*SF); % transform it in "frame"
    if min(abs(J-position)) > limite % is the minimum distance between this postion and each spike is above the limite
        try
            S = single(I(:,:,loca+win(1):loca+win(2)));
            S = imresize(S,r,'nearest'); 
            if c == 1, v.method = 1; v.dc = 1; S = OIA_SA_globalactivity(S,v); end
            if win(1)<0
                M = mean(S(:,:,1:abs(win(1))),3);
            else
                M = mean(S(:,:,1:size(S,3)),3);
            end
            for f = 1:size(S,3)
                S(:,:,f) = 100*(S(:,:,f) - M)./M;
            end
            disp(['random pull ' num2str(n_blank) '/' num2str(size(SM,4)) ' (t=' num2str(position) 's)'])
            SMs(:,:,:,n_blank+1) = S;
            n_blank = n_blank + 1;
            
        catch
            disp(' > out of range')
        end
    else
        disp([' > too close to a pull:' num2str(min(abs(J-position))) 's (so not added...)'])
    end
end

maxM=squeeze(max(max(squeeze(mean(SM,3)),[],1),[],2));
SM = SM(:,:,:,find(maxM~=0));
maxMs=squeeze(max(max(squeeze(mean(SMs,3)),[],1),[],2));
SMs = SMs(:,:,:,find(maxMs~=0));
maxT=squeeze(max(max(squeeze(mean(ST,3)),[],1),[],2));
ST = ST(:,:,:,find(maxT~=0));

if size(SM,4)>size(SMs,4)
    SM = SM(:,:,:,1:size(SMs,4));
elseif size(SMs,4)>size(SM,4)
    SMs = SMs(:,:,:,1:size(SM,4));
end

[XX,YY,ZZ,TT]=size(SM);
SM = SM(:);
loca = find(isnan(SM)==1);
SM(loca) = zeros(size(loca));
SM = reshape(SM,XX,YY,ZZ,TT);
[XX,YY,ZZ,TT]=size(SMs);
SMs = SMs(:);
loca = find(isnan(SMs)==1);
SMs(loca) = zeros(size(loca));
SMs = reshape(SMs,XX,YY,ZZ,TT);

% maxi = min([size(SM,4) size(SMs,4)]);
% SM = SM(:,:,:,1:maxi);
% SMs = SMs(:,:,:,1:maxi);

ST = mean(ST,4);
disp(['kept pull: ' num2str(size(SM,4)) '(/' num2str(length(J)-2) ')x'])