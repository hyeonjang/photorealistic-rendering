%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CS-454 Course Project, Utility Maximization %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% define constants and other usefull stuff

%c and l limited between 0 and 1
clear all;
clc;
echo off
disp( 'Utility Maximization Problem' );
cvx_pause(false);

disp('Computing discount factor for Utility');
t_end=10;
t = 0:t_end-1;

beta = exp(log(0.8)*t); %discount factor

disp('Computing prices');
p = ones(1,length(t))*0.5;

disp('Computing economical growth');
tau = ones(1,length(t))*(-0);

disp('Computing interest rate');
R = 0.8*ones(1,length(t));

c_matrix = [];
l_matrix = [];
lambda_matrix = [];
sigma_matrix = [];
epsilon_matrix = [];
rho_matrix = [];
alpha_matrix = [];


%% solve the problem
steps = 25;

for k=1:steps
    beta = exp(log(20/25).*t); %discount factor
    cvx_begin
        variables c(t_end) l(t_end) s(t_end) m(t_end);
        dual variables lambda{t_end} mu epsilon rho sigma alpha;
        %maximize(sum(beta *(log(c) + log(1-l))))
        maximize(sum(beta * (log(c) + log(1-l))))
        subject to
            epsilon: 0<=c<=1;
            rho: 0<=l<=1;
            m(1)==0.5;
            for k = 1:t_end-1,
                 lambda{k} : m(k+1) == -s(k+1) + m(k) + (1+R(k))*s(k) + p(k)*l(k)+tau(k)-p(k)*c(k);
            end
            sigma: s(t_end)==0;
            m == p'.*c;
    cvx_end
    c_matrix = [c_matrix c];
    l_matrix = [l_matrix l];
    lambda_matrix = [lambda_matrix c];
    sigma_matrix = [sigma_matrix sigma];
    rho_matrix = [rho_matrix rho];
    epsilon_matrix = [epsilon_matrix epsilon];
end


%% Primal variables
k=1:steps;

figure
surf(k*100/steps,t,c_matrix);
title(strcat('Consumption behavior for R=',num2str(R(1))));
xlabel('\beta (discount factor)');
ylabel('t (time step)');
zlabel('Consumed amount');

figure
surf(k*100/steps,t,l_matrix);
title(strcat('Labor behavior for R=',num2str(R(1))));
xlabel('\beta (discount factor)');
ylabel('t (time step)');
zlabel('Ratio of labor');


%% DUAL VARIABLES

figure
surf(k*100/steps,t,lambda_matrix);
title(strcat('Dual variable \lambda for R=',num2str(R(1))));
xlabel('\beta (discount factor)');
ylabel('t (time step)');
zlabel('\lambda');

figure
plot(k*100/steps,sigma_matrix);
title(strcat('Dual variable \sigma for R=',num2str(R(1))));
xlabel('t (time step)');
ylabel('\sigma');

figure
surf(k*100/steps,t,rho_matrix);
title(strcat('Dual variable \rho for R=',num2str(R(1))));
xlabel('\beta (discount factor)');
ylabel('t (time step)');
zlabel('\rho');

figure
surf(k*100/steps,t,epsilon_matrix);
title(strcat('Dual variable \epsilon for R=',num2str(R(1))));
xlabel('\beta (discount factor)');
ylabel('t (time step)');
zlabel('\epsilon');
