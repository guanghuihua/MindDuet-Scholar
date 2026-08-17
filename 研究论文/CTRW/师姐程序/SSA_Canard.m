N = 600;%NxN Grid 
eps = 0.3;%Strength of noise % 0.5 original 
lowx = -3.0;%left end of numerical domain
lowy = -3.0;%bottom end of numerical domain
Sp = 6.0;%length (and height) of the numerical domain
Sample = 10000000;%Sample size of Monte Carlo simulator
dt = 0.001;%time step of numerical ODE scheme in MC simulator
clf
%{
t1=clock;
data1 = MC_2D(lowx, lowy, Sp, N, eps, Sample, dt);%Generate approximate data
t2=clock;
disp('MC_2D data generated');
computer_running_time=etime(t2,t1)
%}
t1=clock;
data2 = SSA_2D(lowx, lowy, Sp, N, eps, Sample);
t2=clock;
disp('SSA_2D data generated');
computer_running_time=etime(t2,t1)


[A, b] = Matrix_2D(lowx, lowy, Sp, N, eps);%Build matrix
disp('matrix built');
b = b - A*data2;
x = lsqminnorm(A,b);%Find the least norm solution
y = x + data2;
V = zeros(N+1,N+1);%Numerical solution
W = zeros(N+1,N+1);%A rough Solution obtained by Monte Carlo
for i = 1:N+1
    for j = 1:N+1
        V(i,j) = V(i,j) + y((i-1)*(N+1) + j);
        W(i,j) = W(i,j) + data2((i-1)*(N+1)+j);
    end
end
subplot(2,1,1)
mesh(W)
subplot(2,1,2)
mesh(V)

function y = canard(x)%ODE
delta = 0.1;
a = 1 - delta/8 - 3*delta^2/32 - 173*delta^3/1024 - 0.01;
y = zeros(2,1);
y(1) = (x(1) + x(2) - x(1)^3/3)/delta;
y(2) = a - x(1);
end

function data1 = MC_2D(lowx, lowy, Sp, N, eps, Sample, dt)
h = Sp/N;
data = zeros(N*N,1);
P = zeros(N,N);
x_old = [1 1]';

%xplot=1;% For plotting
%yplot=1;% For plotting
for i = 1:Sample
    x_new = x_old + dt*canard(x_old) + eps*sqrt(dt)*randn(2,1);
    xx = x_new(1);
    yy = x_new(2);
    x_n = ceil((xx - lowx)/h);
    y_n = ceil((yy - lowy)/h);
    if x_n >= 1 && x_n <=N && y_n >= 1 && y_n <= N
        P(x_n, y_n) = P(x_n,y_n) + 1;%Count number of sample points in each bin
    end
    x_old = x_new;   
    %xplot(i+1) = x_new(1);% For plotting
    %yplot(i+1) = x_new(2);% For plotting
end
for i = 1:N
    for j = 1:N
        data((i-1)*N + j) = P(i,j);
    end
end
data1 = data/(h^2*sum(data));%Normalization
fprintf('Sample Size = %d\n',Sample);
%{
subplot(2,2,1)
plot(xplot,yplot)
%}
subplot(2,1,1)
P=P/(h^2*sum(data));
mesh(P)

end

function data2 = SSA_2D(lowx, lowy, Sp, N, eps, Sample)
%%[dX,dY]=[1/epsilon*(Y-1/3*X^3+X),(a-X)]dt+sigma*dW, 
% dY=mu(Y)dt +sqrt(2)sigma(Y)dW in (2.1.2) in Vanden-Eijnden's paper
% mu(x,y)=[10*(Y-1/3*X^3+X),(0.9964-X)],and
% M_{1,1}=sigma^2/2, M_{1,2}=0, M_{2,1}=0, M_{2,2}=sigma^2/2, 
%--------------------------------------------------------------------------

h = Sp/N;%delta_x=delta_x_plus=delta_x_minus=h;%delta_y=delta_y_plus=delta_y_minus=h;
data = zeros((N+1)*(N+1),1);
P = zeros(N+1,N+1);
x=1;y=1;%initial value
t=0;%Time variable
%{
tplot=0;
xplot=x;
yplot=y;% For plotting
%}
t_stop=50000;    
n=0; %Reaction counter.
delta = 0.1;
a = 1 - delta/8 - 3*delta^2/32 - 173*delta^3/1024 - 0.01;
sigma=eps;
M_11=sigma^2/2; M_22=sigma^2/2;
%% STEP 1
C1=(M_11/h^2); C2=(M_22/h^2);
while t<t_stop
    mu_1=(y-1/3*x^3+x)/delta;%Q_u
    mu_2=a-x;%Q_u
    q1=max(mu_1,0)/h+C1;%Q_u
    q2=-min(mu_1,0)/h+C1;%Q_u
    q3=max(mu_2,0)/h+C2;%Q_u
    q4=-min(mu_2,0)/h+C2;%Q_u
    %{
    mu_tilde_1=20/sigma^2*(y-1/3*x^3+x);%Q_c
    mu_tilde_2=2/sigma^2*(a-x);%Q_c
    q1=exp(1/2*mu_tilde_1*h)*C1; % X_{i,j} -> X_{i+1,j)%Q_c
    q2=exp(-1/2*mu_tilde_1*h)*C1; % X_{i,j} -> X_{i-1,j)%Q_c
    q3=exp(1/2*mu_tilde_2*h)*C2; % X_{i,j} -> X_{i,j+1)%Q_c
    q4=exp(-1/2*mu_tilde_2*h)*C2; % X_{i,j} -> X_{i,j-1)%Q_c
    %}
    lambda=q1+q2+q3+q4; % four channels
    %% STEP 2
    r = rand(2,1);
    tau = -log(r(1))/lambda;% 
    mu_number = sum(r(2)*lambda<=cumsum([q1,q2,q3,q4]));
    %% STEP 3
    t = t + tau;
    % Adjust population levels.
    switch mu_number
        case 4
            x = x + h; % X_{i,j} -> X_{i+1,j)
        case 3
            x = x - h; % X_{i,j} -> X_{i-1,j)
        case 2
            y = y + h; % X_{i,j} -> X_{i,j+1)
        case 1
            y = y - h; % X_{i,j} -> X_{i,j-1)
    end
    x_n = round((x - lowx)/h);
    y_n = round((y - lowy)/h);
    if x_n >= 1 && x_n <=N+1 && y_n >= 1 && y_n <= N+1
        P(x_n, y_n) = P(x_n,y_n) + 1;%Count number of sample points in each bin
    end
    n = n + 1;
    if n==Sample
        break
    end
    %{
    tplot(n+1) = t;
    xplot(n+1) = x;
    yplot(n+1) = y;
    %}
end
    for i = 1:N+1
    for j = 1:N+1
        data((i-1)*(N+1) + j) = P(i,j);
    end
    end
    data2 = data/(h^2*sum(data));%Normalization
    fprintf('Sample Size = %d\n',n);
    %{
    subplot(2,2,3)
    plot(xplot,yplot)
    %}
    P=P/(h^2*sum(data));
    subplot(2,1,2)
    mesh(P)
    %plot3(tplot,xplot,yplot)
    
end

function [A,b]= Matrix_2D(lowx, lowy, Sp, N, eps0)
eps = eps0^2/2;%Coefficient of the Laplacian
h = Sp/N;%grid size
subindex = @(vec,index) vec(index);
f1 = @(x,y) subindex(canard([x y]'),1);%x component of the vector field
f2 = @(x,y) subindex(canard([x y]'),2);%y component of the vector field
%{
b = zeros((N-2)*(N-2) + 1,1);
b((N-2)*(N-2) + 1) = 1/h^2;
Num = 5*(N-2)*(N-2) + N*N;
%}
b = zeros((N-1)*(N-1) + 1,1);
b((N-1)*(N-1) + 1) = 1/h^2;
Num = 5*(N-1)*(N-1) + (N+1)*(N+1);
index1 = zeros(1,Num);
index2 = zeros(1,Num);
value = zeros(1,Num);
count = 1;
%{
for i = 2:N-1
    for j = 2:N-1
        xx = (i-1)*h + lowx ;%Important: A grid point is in the middle of a Monte Carlo bin
        yy = (j-1)*h + lowy ;
        index1(count) = (i-2)*(N-2) + j - 1;
        index2(count) = (i-1)*N + N + j;
        value(count) = -f1(xx+h,yy)/(2*h) + eps/(h^2);
        count = count +1;
        

        index1(count) = (i-2)*(N-2) + j - 1;
        index2(count) = (i-1)*N - N +j;
        value(count) = f1(xx-h,yy)/(2*h) + eps/(h^2);
        count = count +1;


        index1(count) = (i-2)*(N-2) + j - 1;
        index2(count) = (i-1)*N+j+1;
        value(count) = -f2(xx,yy+h)/(2*h) + eps/(h^2);
        count = count +1;

        
        index1(count) = (i-2)*(N-2) + j - 1;
        index2(count) = (i-1)*N + j -1;
        value(count) = f2(xx,yy-h)/(2*h) + eps/(h^2);
        count = count +1;

        
        index1(count) = (i-2)*(N-2) + j - 1;
        index2(count) = (i-1)*N+j;
        value(count) = -4*eps/(h^2);
        count = count +1;
    end
end
%}
for i = 2:N
    for j = 2:N
        xx = (i-1)*h + lowx ;%Important: A grid point is in the middle of a Monte Carlo bin
        yy = (j-1)*h + lowy ;
        index1(count) = (i-2)*(N-1) + j - 1;
        index2(count) = (i-1)*(N+1) + (N+1) + j;
        value(count) = -f1(xx+h,yy)/(2*h) + eps/(h^2);
        count = count +1;
        
        index1(count) = (i-2)*(N-1) + j - 1;
        index2(count) = (i-1)*(N+1) - (N+1) +j;
        value(count) = f1(xx-h,yy)/(2*h) + eps/(h^2);
        count = count +1;

        index1(count) = (i-2)*(N-1) + j - 1;
        index2(count) = (i-1)*(N+1)+j+1;
        value(count) = -f2(xx,yy+h)/(2*h) + eps/(h^2);
        count = count +1;

        index1(count) = (i-2)*(N-1) + j - 1;
        index2(count) = (i-1)*(N+1) + j -1;
        value(count) = f2(xx,yy-h)/(2*h) + eps/(h^2);
        count = count +1;
     
        index1(count) = (i-2)*(N-1) + j - 1;
        index2(count) = (i-1)*(N+1)+j;
        value(count) = -4*eps/(h^2);
        count = count +1;
    end
end
%{
for i = 1:N
    for j = 1:N
        index1(count) = (N-2)*(N-2) + 1;
        index2(count) = (i-1)*N + j;
        value(count) = 1;
        count = count + 1;
    end
end
%}
for i = 1:N+1
    for j = 1:N+1
        index1(count) = (N-1)*(N-1) + 1;
        index2(count) = (i-1)*(N+1) + j;
        value(count) = 1;
        count = count + 1;
    end
end
%A = sparse(index1,index2,value,(N-2)*(N-2) + 1,N*N);%Use sparse matrix
A = sparse(index1,index2,value,(N-1)*(N-1) + 1,(N+1)*(N+1));%Use sparse matrix
end

