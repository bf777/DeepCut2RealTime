function [ output ] = find_index( input )
%UNTITLED2 Summary of this function goes here
%   input is a cell
% output is a vector
len1=size(input,1);
tem_index=[];
for i=1:len1
    tem1=input(i).name;
    tem1=tem1(6:end-4);
    tem_index(i)=str2num(tem1);
end
output=tem_index;
end

