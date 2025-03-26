
%% create the stock and call payoff vectors
stock = [1:100]';
for i = [1:100]
    for j = [1:100]
        call(i,j) = max(stock(j)-5*i,0);
        put(i,j) = max(5*i - stock(j),0);
    end
end
 

Import portfolio profile
data = readtable('Portfolio Surgery Test 1.xlsx');
portfolio = data{:, 2}; % Second column for "Portfolio 1"

 
%% see portfolio payoff
plot(stock,portfolio)
 
%% transpose
call = call';
put = put';
 
%% Calls?
plot(call);
 
%% puts??
plot(put);
%% create x vector
x = [call put];
 
%% take the negative piece of portfolio
negativeportfolio =min(portfolio,0);
 
%negativeportfolio = negativeportfolio';
 
%%
b = regress(negativeportfolio, x);
%%
aftersurgery = portfolio - x*b;
 
%% plot and check!
 
plot(stock,portfolio);
hold on;
plot(stock, aftersurgery);
 
%% see piecewise linear approximation
 
optionpayoff = x*b;
plot(stock([1:100]), portfolio([1:100]));
hold on
plot(stock([1:100]),optionpayoff([1:100]));
hold off

