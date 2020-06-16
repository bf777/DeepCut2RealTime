%% analysis of real-time tracking data by Dongsheng Xiao
clear all, close all
X = 128; % pixel size of the data
% directory = 'E:\real-time-tracking\led_testing\JS1_200Hz_2\';
% date = '2019-10-11';
date = 'led_testing';
animal = 'DS33';
directory = ['E:\real-time-tracking\led_testing\' animal '_200Hz\'];
resu = dir(directory);ind = 0;
%get triggers
B = 800; %start of stable frame name % 800
for i = 5
    namefile = resu(i).name(length(resu(i).name)-2:length(resu(i).name));
    if  namefile=='csv'
        file = [directory resu(i).name];
        ind = ind + 1;
        disp(file);
        T = csvread(file,1,0);
    end;
    FrameTime = T(:,2);
    ThresholdTime = T(:,5);
    sp = ThresholdTime(B+1:size(FrameTime,1),1);
    sp(1)=FrameTime(B+1);
    sp(size(sp,1))=FrameTime(size(FrameTime,1));
    duration = sp(size(sp,1))-sp(1);
    sp(sp==0)=[];
    data(ind).sp = sp;
end
%get frames
A = size(resu,1);
a = dir([directory '/*.png']);
file_name_index=find_index(a);
PNG = size(a,1);
F = size(FrameTime,1)-B;
%
for i = B
    temi=find(file_name_index==i);
    file = [directory a(temi).name];
    info = imfinfo(file); tampon = imread(file);
    I = zeros(size(tampon,1),size(tampon,2),F,'uint16');
end
for i = B:max(file_name_index)
    try
        temi=find(file_name_index==i);
        file = [directory a(temi).name];
        info = imfinfo(file); 
        tampon = rgb2gray(imread(file));
        I(:,:,i+1-B) = tampon(:,:);
    catch
        continue;
    end
end
%
s = squeeze(sum(sum(I,1),2)); plot(s), [x,y]=ginput(1);
s = bwlabel(s>y); I = I(:,:,(s==1)); close all
fileroi = [file(1:length(file)-4) '_roi.mat']; resu2 = dir(fileroi);
if length(resu2)==0, roid = OIA_n(single(I(:,:,B+1))).^.5; roi = OIA_roi('martin',roid,roid,1);
    close all, else load(fileroi); end
r = X/size(I,1); if r ~= 1, I = imresize(I,r,'nearest'); roi = imresize(roi,r,'nearest'); end;
SF = size(I,3)/duration;
lum = squeeze(mean(mean(I,1),2));
subplot(131), imagesc(roi);
subplot(132), imagesc(mean(I,3)); colormap gray
title(['number of ePhys channels: ' num2str(length(data)) ', imaging freq: ' num2str(SF) 'Hz'])
subplot(133), plot(lum);

%% 3. average with trigger
clc
win = [-0.55 0.55]; % temporal window (how many sec before and after trig)
refper = [0.0 1000000]; % refractory period [min max] (sec)
winA =[1 120];% sec
If = I;
J = data(1).sp - data(1).sp(1);
[SM,SMs,ST] = OIA_pull_average(If,J,SF,refper,0,-1,win,winA);
size(If)
size(J)

%% 3. check signal
method =1; % 1 = maxDF/F, 2=Z-score, 3=STD, 4=min(DF/F), 5=negative Z-score, 6=min/max DF/F combined
sub = 1; % =1 if substract shuffle
curwin = [-0.5 0.5]; % window to generate the map (s)
k=1; % roi size (for click)
display_option = 2; % display line or regions... just cosmetic...
%exportfile = ['C:\Users\user\Desktop\led_testing_stats\' date '_' animal '_200Hz_4_train.txt'];
exportfile = ['C:\Users\user\Desktop\led_testing_stats\' animal '_200Hz_LED_testing_TEST.txt'];
ksd = +4; % to calculate onset (k*SD) %4
map = OIA_pull_map(SM,SMs,roi,SF,method,sub,curwin,win);
x=0; y=0;
for i = 1:1000
    subplot(121), imshow(map), colormap jet
    hold on, plot(x,y,'wo'); plot(x,y,'xk'); hold off
    [x,y]=myginput;
    s = squeeze(mean(mean(SM(y-k:y+k,x-k:x+k,:,:),1),2));
    s_s = squeeze(mean(mean(SMs(y-k:y+k,x-k:x+k,:,:),1),2));
    ms = mean(s,2);
    ss = std(s,[],2)./sqrt(size(s,2));
    
    mss = mean(s_s,2);
    sss = std(s_s,[],2)./sqrt(size(s_s,2));
    
    tt = ([1:length(ss)]'-round(abs(win(1)).*SF)-1)/SF;
    subplot(122),
    
    if display_option == 1
        plot(tt,ms,'r','LineWidth',5); hold on, plot(tt,ms-ss,'r'); plot(tt,ms+ss,'r');
    else
        xx = [tt' fliplr(tt')];
        yy = [ms'+ss' fliplr(ms'-ss')];
        fill(xx,yy,[0.68 .92 1],'EdgeAlpha',0,'FaceAlpha',.5);  hold on
        plot(tt,ms,'Color',[0 .45 .74]);
        
        pre = round(-win(1)*SF);
        mmm = ms - mss;
        sd = std(mss);
        lat = (min(find(ms>(ksd*sd)))-1-pre)/SF;
        try dlmwrite(exportfile,[tt,ms, ss, (find(ms>(ksd*sd))-1-pre/SF)],'\t'); end
        mmmi= ms/max(ms);
        t2p = (min(find(mmmi==1))-1-pre)/SF;
        mmmi = mmmi .* (mmmi>=0);
        sumi = sum(mmmi);
        disp('----------------------------------------')
        disp(['onset (>' num2str(ksd) '*STD) = ' num2str(1000*lat) 'ms'])
        disp(['t2p = ' num2str(1000*t2p) 'ms'])
        disp(['peak = ' num2str(max(ms)) '%'])
        disp(['area(norm) = ' num2str(sumi)])
        stats_data_name = strcat('C:\Users\user\Desktop\led_testing_stats\', ...
            'LED_data_table_', date, '_', animal, '_testing_TEST.csv');
        stats_table = cell2table({date, animal, length(J), (ksd), (1000*lat), ...
        (1000*t2p), (max(ms)), (sumi)});
        stats_table.Properties.VariableNames = {'date' 'animal' 'triggers' 'ksd' 'onset' ...
            'time2peak' 'peak' 'area_norm'};
        writetable(stats_table, stats_data_name);
        disp('Table saved!');
    end
    %     plot(tt,ms-mss,'b','LineWidth',2)
    plot([min(tt) max(tt)],[0 0],'k')
    range = 1*(max(ms)-min(ms));
    plot([0 0],[min(ms)-range max(ms)+range],'k')
    
    hold off
    grid off, xlabel('time (s)');
    %     grid off, xlabel('time (s)'); ylabel('DF/F (%)')
    title([num2str(size(SM,4)) ' triggers'])
    
    hold on,
    try plot([lat lat],[min(mmm) max(mmm)],'r'); plot([win(1) win(2)],[ksd*sd ksd*sd],'r'); end
    %     try, plot([lat2 lat2],[min(mmm) max(mmm)],'m'); plot([win(1) win(2)],-[ksd2*sd ksd2*sd],'m'); end
    hold off
end

%% check triggers and LED for this location
bin = 30; % bin in ms
s= squeeze(mean(mean(If(y-k:y+k,x-k:x+k,:),1),2));
s = 100*(s - mean(s))./mean(s);
t_s = [1:length(s)]'/SF;
bin = bin/1000;
N = floor(max(t_s)/bin);
clear t_f fr
for i = 1:N
    fr(i) = sum((J((J>=((i-1)*bin))&(J<(i*bin))))>0);
    t_f(i) = i*bin;
end
barre = [min(s) min(s)-1];
subplot(111)
hold on
plot(t_s,s,'r'), hold on
for i = 2:length(J)-1,
    plot([J(i) J(i)],barre,'b');
end;
%   plot(t_f,10*fr+max(s),'g'),
hold off
xlabel('time (s)')
legend('LED','trigger')
% 'firerate*10'