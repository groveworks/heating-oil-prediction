using Plots
using DataFrames
using DelimitedFiles
using Dates
using Flux
using BSON: @save

"""
Load CSV data into a DataFrame.
"""
function read_csv(filename::String, delimiter::Char=',')
    data, headers = readdlm(filename, delimiter, header=true)
    df = DataFrame(data, vec(headers))
    return df
end

df = read_csv("data/heating oil prices.csv")


# for the weekly price of dollars per gallon columns
df[!, :"weekly__dollars_per_gallon"] = convert.(Float32, df[!, :"weekly__dollars_per_gallon"])

# Convert string columns to the date types
df[!, :"\ufeffdate"] = Date.(df[!, :"\ufeffdate"], Dates.DateFormat("u d, yyyy"))
df[!,:"years"] = year.(df[!,:"\ufeffdate"])
df[!,:"months"] = month.(df[!,:"\ufeffdate"])
df[!,:"weeks"] = week.(df[!,:"\ufeffdate"])

df[!, :years] = convert.(Float32, df[!, :years])
df[!, :months] = convert.(Float32, df[!, :months])
df[!, :weeks] = convert.(Float32, df[!, :weeks])

X_train = Array(first(df[!, [:years,:months,:weeks]], Int(nrow(df)*0.75)))'
X_test = last(df[!, [:years,:months,:weeks]], Int(nrow(df)*0.25))
y_train = Array(first(df[!, :weekly__dollars_per_gallon], Int(nrow(df)*0.75)))'
y_test = last(df[!, :weekly__dollars_per_gallon], Int(nrow(df)*0.25))
train_loader = Flux.Data.DataLoader((X_train, y_train), shuffle=true);
test_loader = Flux.Data.DataLoader((data=X_test, label=y_test), batchsize=64, shuffle=true);
first(train_loader, 10)
model = GRU(3 => 1)
#model = Dense(3 => 1)
print(model)
optim = Flux.setup(Adam(0.01), model)
losses = []
for epoch in 1:10
    for (x,y) in train_loader
        loss, grads = Flux.withgradient(model) do m
            # Evaluate model and loss inside gradient context:
            y_hat = m(x)
            Flux.mae(y_hat, y)
        end
        Flux.update!(optim, model, grads[1])
        push!(losses, loss)
    end
end

@save "../mymodel.bson" model
