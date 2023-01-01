using DataFrames
using DelimitedFiles

"""
Load CSV data into a DataFrame.
"""
function read_csv(filename::String, delimiter::Char=',')
    data, headers = readdlm(filename, delimiter, header=true)
    df = DataFrame(data, vec(headers))
    return df
end

