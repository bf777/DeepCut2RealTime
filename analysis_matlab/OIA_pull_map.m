function [map,mapmap,LUT_range] = OIA_pull_map(SM,SMs,roi,SF,method,sub,curwin,win)
% function [map,map_real,LUT_range] = OIA_pull_map(SM,SMs,roi,SF,method,sub,curwin,win)
% roi = roi map
% SM = [X,Y,F,T] of pull
% SMs = idem but no pull (random)
% SF : sampling freq (Hz)
% method 1 = maxDF/F, 2=Z-score, 3=STD, 4=min(DF/F), 5=negative Z-score,
% 6=min/max DF/F combined, 7=French colormap 
% sub =1 if substract shuffle
% curwin = window to generate the map (s), eg. [-1 1] 
% win = temporal window (how many sec before and after trig), eg: [-5 5]

if method < 0, real_norm = 1; else, real_norm = 0; end
method = abs(method);
win = round(abs(win).*SF);
SMM = mean(SM,4);
SMMs = mean(SMs,4);

curwin = round(curwin*SF);

if method == 1
    map = max(SMM(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3); % DF/F
    maps = max(SMMs(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3); % DF/F
elseif method == 2
    map = max(SMM(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3) ./ std(SMM(:,:,1:win(1)),[],3); % Z
    maps = max(SMMs(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3) ./ std(SMMs(:,:,1:win(1)),[],3);
elseif method == 3
    map = std(SMM,[],3) - std(SMM(:,:,1:win(1)),[],3); % STD
    maps = std(SMMs,[],3) - std(SMMs(:,:,1:win(1)),[],3); % STD
elseif method == 4
    map = min(SMM(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3); % DF/F
    maps = min(SMMs(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3); % DF/F    
elseif method == 5
    map = -min(SMM(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3) ./ std(SMM(:,:,1:win(1)),[],3); % Z
    maps = -min(SMMs(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3) ./ std(SMMs(:,:,1:win(1)),[],3);
elseif ((method == 6)|(method==7))
    map = max(SMM(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3); % DF/F
    maps = max(SMMs(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3); % DF/F
    map2 = -min(SMM(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3); % DF/F
    maps2 = -min(SMMs(:,:,win(1)+curwin(1):win(1)+curwin(2)),[],3); % DF/F       
else
    map = mean(I,3);
    maps = 0;
end;
if sub == 1, 
    map = map-maps;
    if ((method == 6)|(method==7))
        map2 = map2-maps2;
    end
end
mapmap = map;
if ((method==1)|(method==2))
    map = map .*(map>0);
end
if (method==4)
    map = map .*(map<0);
end
if ((method==6)|(method==7))
    map = map .*(map>0);
    map2 = map2 .*(map2>0);
end

%disp(['range: ' num2str(min(map(:))) '-' num2str(max(map(:)))])

roimap = imresize(roi,size(map));
if method == 6
    mapraw = map; 
    mapraw2 = map2;
    mapraw = mapraw .* single(roimap);
    mapraw2 = mapraw2 .* single(roimap);
%     map = OIA_pttt('data',map,1,'borne',[min(map(roimap==1)) max(map(roimap==1))]);
%     map2 = OIA_pttt('data',map2,1,'borne',[min(map2(roimap==1)) max(map2(roimap==1))]);
%     
    mini = min(map(roimap==1));
    maxi = max(map(roimap==1));
    mini2 = min(map2(roimap==1));
    maxi2 = max(map2(roimap==1));
    map = map.*((map>mini)&(map<maxi)) + mini.*(map<=mini) + maxi.*(map>=maxi);
    map2 = map2.*((map2>mini2)&(map2<maxi2)) + mini2.*(map2<=mini2) + maxi2.*(map2>=maxi2);
    
    LUT_range = [-max(map2(:)) max(map(:))];
    disp([num2str(LUT_range(1)) ' to ' num2str(LUT_range(2)) ' %']) 
    %map = map - min(map(:));
    %map2 = map2 - min(map2(:));
    
    if real_norm == 1
        map = map / max(map(:));
        map2 = map2 / max(map2(:));        
    else
        maximaxi = max([max(map(:)) max(map2(:))]);
        map = map / maximaxi;
        map2 = map2 / maximaxi;
    end
    
    RGB = zeros(size(map,1),size(map,2),3);
    RGB(:,:,1) = 255*map .* roimap;
    RGB(:,:,2) = 255*map2 .* roimap;
   
    map = uint8(RGB);
elseif method == 7
    mapraw = map; 
    mapraw2 = map2;
    mapraw = mapraw .* single(roimap);
    mapraw2 = mapraw2 .* single(roimap);
    mini = min(map(roimap==1));
    maxi = max(map(roimap==1));
    mini2 = min(map2(roimap==1));
    maxi2 = max(map2(roimap==1));
    map = map.*((map>mini)&(map<maxi)) + mini.*(map<=mini) + maxi.*(map>=maxi);
    map2 = map2.*((map2>mini2)&(map2<maxi2)) + mini2.*(map2<=mini2) + maxi2.*(map2>=maxi2);
    
    LUT_range = [-max(map2(:)) max(map(:))];
    disp([num2str(LUT_range(1)) ' to ' num2str(LUT_range(2)) ' %'])   
%     map = map - min(map(:));
%     map2 = map2 - min(map2(:));
    
    if real_norm == 1
        map = map / max(map(:));
        map2 = map2 / max(map2(:));
    else
        maximaxi = max([max(map(:)) max(map2(:))]);
        map = map / maximaxi;
        map2 = map2 / maximaxi;
    end
    RGB = 255*((1-map).*(1-map2));
    RGB = cat(3,RGB,RGB,RGB);
    RGB(:,:,1) = RGB(:,:,1) + 255.*map;
    RGB(:,:,3) = RGB(:,:,3) + 255.*map2;
    map = uint8(RGB);
    nana = sum(map,3)==0;
    for i = 1:3
        map(:,:,i) = map(:,:,i) + 255*uint8(nana);
    end
    
        
else
    
    LUT_range = [min(map(:)) max(map(:))];
    disp(['RANGE: ' num2str(LUT_range(1)) ' - ' num2str(LUT_range(2))])
    
    mapraw = map; mapraw = mapraw .* single(roimap);
    map = OIA_pttt('data',map,1,'borne',[min(map(roimap==1)) max(map(roimap==1))]);
    map = OIA_colorroi(map,jet,roi);
end
