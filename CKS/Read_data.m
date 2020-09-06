function A= Read_data(data)
data='Data2.xlsx';
Tbl = readtable(data);
%Tbl=rmmissing(Tbl); 
VarNames = Tbl.Properties.VariableNames;                                 % Recover Variable Names (Column Titles)
VarInputs=cell(6,1);
VarInputs{1,1}='GRADE';
VarInputs{2,1}='BDFCRG';
VarInputs{3,1}='HQCRG';
VarInputs{4,1}='SCCRG';
VarInputs{5,1}='BDFADD';
VarInputs{6,1}='HQADD';

for ii=1:6
ColIdx = find(strcmp(VarNames, VarInputs{ii,1})); 
use=Tbl(:,ColIdx);
use2=use{:,:};
Inns(:,ii)=use2;
end

%% 
Varoutputs=cell(7,1);
Varoutputs{1,1}='BlackSpeck__SFL_';
Varoutputs{2,1}='Colour_a_value';
Varoutputs{3,1}='Colour_b_value';
Varoutputs{4,1}='Colour_L_value';
Varoutputs{5,1}='DPSContent';
Varoutputs{6,1}='Melt_Viscosity';
Varoutputs{7,1}='Mositure_Content_p';
for ij=1:7
ColIdxx = find(strcmp(VarNames, Varoutputs{ij,1})); 
usea=Tbl(:,ColIdxx);
useb1=usea{:,:};
outts(:,ij)=useb1;
end

A=[Inns outts];
A=rmmissing(A); 
end