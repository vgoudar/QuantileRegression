ў 
сА
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Ѕ
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
С
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.12.02v2.12.0-rc1-12-g0db597d0d758ф
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_3
[
count_3/Read/ReadVariableOpReadVariableOpcount_3*
_output_shapes
: *
dtype0
b
total_3VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_3
[
total_3/Read/ReadVariableOpReadVariableOptotal_3*
_output_shapes
: *
dtype0
b
count_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_4
[
count_4/Read/ReadVariableOpReadVariableOpcount_4*
_output_shapes
: *
dtype0
b
total_4VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_4
[
total_4/Read/ReadVariableOpReadVariableOptotal_4*
_output_shapes
: *
dtype0
b
count_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_5
[
count_5/Read/ReadVariableOpReadVariableOpcount_5*
_output_shapes
: *
dtype0
b
total_5VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_5
[
total_5/Read/ReadVariableOpReadVariableOptotal_5*
_output_shapes
: *
dtype0
b
count_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_6
[
count_6/Read/ReadVariableOpReadVariableOpcount_6*
_output_shapes
: *
dtype0
b
total_6VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_6
[
total_6/Read/ReadVariableOpReadVariableOptotal_6*
_output_shapes
: *
dtype0
b
count_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_7
[
count_7/Read/ReadVariableOpReadVariableOpcount_7*
_output_shapes
: *
dtype0
b
total_7VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_7
[
total_7/Read/ReadVariableOpReadVariableOptotal_7*
_output_shapes
: *
dtype0
b
count_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_8
[
count_8/Read/ReadVariableOpReadVariableOpcount_8*
_output_shapes
: *
dtype0
b
total_8VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_8
[
total_8/Read/ReadVariableOpReadVariableOptotal_8*
_output_shapes
: *
dtype0
b
count_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_9
[
count_9/Read/ReadVariableOpReadVariableOpcount_9*
_output_shapes
: *
dtype0
b
total_9VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_9
[
total_9/Read/ReadVariableOpReadVariableOptotal_9*
_output_shapes
: *
dtype0

Adam/v/quantile_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_8/bias
}
*Adam/v/quantile_8/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_8/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_8/bias
}
*Adam/m/quantile_8/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_8/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_8/kernel

,Adam/v/quantile_8/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_8/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_8/kernel

,Adam/m/quantile_8/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_8/kernel*
_output_shapes
:	р]*
dtype0

Adam/v/quantile_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_7/bias
}
*Adam/v/quantile_7/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_7/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_7/bias
}
*Adam/m/quantile_7/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_7/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_7/kernel

,Adam/v/quantile_7/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_7/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_7/kernel

,Adam/m/quantile_7/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_7/kernel*
_output_shapes
:	р]*
dtype0

Adam/v/quantile_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_6/bias
}
*Adam/v/quantile_6/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_6/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_6/bias
}
*Adam/m/quantile_6/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_6/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_6/kernel

,Adam/v/quantile_6/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_6/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_6/kernel

,Adam/m/quantile_6/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_6/kernel*
_output_shapes
:	р]*
dtype0

Adam/v/quantile_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_5/bias
}
*Adam/v/quantile_5/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_5/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_5/bias
}
*Adam/m/quantile_5/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_5/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_5/kernel

,Adam/v/quantile_5/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_5/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_5/kernel

,Adam/m/quantile_5/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_5/kernel*
_output_shapes
:	р]*
dtype0

Adam/v/quantile_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_4/bias
}
*Adam/v/quantile_4/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_4/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_4/bias
}
*Adam/m/quantile_4/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_4/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_4/kernel

,Adam/v/quantile_4/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_4/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_4/kernel

,Adam/m/quantile_4/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_4/kernel*
_output_shapes
:	р]*
dtype0

Adam/v/quantile_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_3/bias
}
*Adam/v/quantile_3/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_3/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_3/bias
}
*Adam/m/quantile_3/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_3/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_3/kernel

,Adam/v/quantile_3/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_3/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_3/kernel

,Adam/m/quantile_3/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_3/kernel*
_output_shapes
:	р]*
dtype0

Adam/v/quantile_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_2/bias
}
*Adam/v/quantile_2/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_2/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_2/bias
}
*Adam/m/quantile_2/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_2/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_2/kernel

,Adam/v/quantile_2/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_2/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_2/kernel

,Adam/m/quantile_2/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_2/kernel*
_output_shapes
:	р]*
dtype0

Adam/v/quantile_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_1/bias
}
*Adam/v/quantile_1/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_1/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_1/bias
}
*Adam/m/quantile_1/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_1/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_1/kernel

,Adam/v/quantile_1/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_1/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_1/kernel

,Adam/m/quantile_1/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_1/kernel*
_output_shapes
:	р]*
dtype0

Adam/v/quantile_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/v/quantile_0/bias
}
*Adam/v/quantile_0/bias/Read/ReadVariableOpReadVariableOpAdam/v/quantile_0/bias*
_output_shapes
:*
dtype0

Adam/m/quantile_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/m/quantile_0/bias
}
*Adam/m/quantile_0/bias/Read/ReadVariableOpReadVariableOpAdam/m/quantile_0/bias*
_output_shapes
:*
dtype0

Adam/v/quantile_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/v/quantile_0/kernel

,Adam/v/quantile_0/kernel/Read/ReadVariableOpReadVariableOpAdam/v/quantile_0/kernel*
_output_shapes
:	р]*
dtype0

Adam/m/quantile_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*)
shared_nameAdam/m/quantile_0/kernel

,Adam/m/quantile_0/kernel/Read/ReadVariableOpReadVariableOpAdam/m/quantile_0/kernel*
_output_shapes
:	р]*
dtype0
{
Adam/v/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:р]*"
shared_nameAdam/v/dense/bias
t
%Adam/v/dense/bias/Read/ReadVariableOpReadVariableOpAdam/v/dense/bias*
_output_shapes	
:р]*
dtype0
{
Adam/m/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:р]*"
shared_nameAdam/m/dense/bias
t
%Adam/m/dense/bias/Read/ReadVariableOpReadVariableOpAdam/m/dense/bias*
_output_shapes	
:р]*
dtype0

Adam/v/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	1р]*$
shared_nameAdam/v/dense/kernel
|
'Adam/v/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/v/dense/kernel*
_output_shapes
:	1р]*
dtype0

Adam/m/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	1р]*$
shared_nameAdam/m/dense/kernel
|
'Adam/m/dense/kernel/Read/ReadVariableOpReadVariableOpAdam/m/dense/kernel*
_output_shapes
:	1р]*
dtype0

Adam/v/embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/v/embedding_6/embeddings

1Adam/v/embedding_6/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_6/embeddings*
_output_shapes

:*
dtype0

Adam/m/embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/m/embedding_6/embeddings

1Adam/m/embedding_6/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_6/embeddings*
_output_shapes

:*
dtype0

Adam/v/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*.
shared_nameAdam/v/embedding_5/embeddings

1Adam/v/embedding_5/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_5/embeddings*
_output_shapes

:8*
dtype0

Adam/m/embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*.
shared_nameAdam/m/embedding_5/embeddings

1Adam/m/embedding_5/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_5/embeddings*
_output_shapes

:8*
dtype0

Adam/v/embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/v/embedding_4/embeddings

1Adam/v/embedding_4/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_4/embeddings*
_output_shapes

:*
dtype0

Adam/m/embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/m/embedding_4/embeddings

1Adam/m/embedding_4/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_4/embeddings*
_output_shapes

:*
dtype0

Adam/v/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/v/embedding_3/embeddings

1Adam/v/embedding_3/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_3/embeddings*
_output_shapes

:*
dtype0

Adam/m/embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/m/embedding_3/embeddings

1Adam/m/embedding_3/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_3/embeddings*
_output_shapes

:*
dtype0

Adam/v/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/v/embedding_2/embeddings

1Adam/v/embedding_2/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_2/embeddings*
_output_shapes

:*
dtype0

Adam/m/embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/m/embedding_2/embeddings

1Adam/m/embedding_2/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_2/embeddings*
_output_shapes

:*
dtype0

Adam/v/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/v/embedding_1/embeddings

1Adam/v/embedding_1/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding_1/embeddings*
_output_shapes

:*
dtype0

Adam/m/embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*.
shared_nameAdam/m/embedding_1/embeddings

1Adam/m/embedding_1/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding_1/embeddings*
_output_shapes

:*
dtype0

Adam/v/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameAdam/v/embedding/embeddings

/Adam/v/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/v/embedding/embeddings*
_output_shapes

:*
dtype0

Adam/m/embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameAdam/m/embedding/embeddings

/Adam/m/embedding/embeddings/Read/ReadVariableOpReadVariableOpAdam/m/embedding/embeddings*
_output_shapes

:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
v
quantile_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_8/bias
o
#quantile_8/bias/Read/ReadVariableOpReadVariableOpquantile_8/bias*
_output_shapes
:*
dtype0

quantile_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_8/kernel
x
%quantile_8/kernel/Read/ReadVariableOpReadVariableOpquantile_8/kernel*
_output_shapes
:	р]*
dtype0
v
quantile_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_7/bias
o
#quantile_7/bias/Read/ReadVariableOpReadVariableOpquantile_7/bias*
_output_shapes
:*
dtype0

quantile_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_7/kernel
x
%quantile_7/kernel/Read/ReadVariableOpReadVariableOpquantile_7/kernel*
_output_shapes
:	р]*
dtype0
v
quantile_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_6/bias
o
#quantile_6/bias/Read/ReadVariableOpReadVariableOpquantile_6/bias*
_output_shapes
:*
dtype0

quantile_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_6/kernel
x
%quantile_6/kernel/Read/ReadVariableOpReadVariableOpquantile_6/kernel*
_output_shapes
:	р]*
dtype0
v
quantile_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_5/bias
o
#quantile_5/bias/Read/ReadVariableOpReadVariableOpquantile_5/bias*
_output_shapes
:*
dtype0

quantile_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_5/kernel
x
%quantile_5/kernel/Read/ReadVariableOpReadVariableOpquantile_5/kernel*
_output_shapes
:	р]*
dtype0
v
quantile_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_4/bias
o
#quantile_4/bias/Read/ReadVariableOpReadVariableOpquantile_4/bias*
_output_shapes
:*
dtype0

quantile_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_4/kernel
x
%quantile_4/kernel/Read/ReadVariableOpReadVariableOpquantile_4/kernel*
_output_shapes
:	р]*
dtype0
v
quantile_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_3/bias
o
#quantile_3/bias/Read/ReadVariableOpReadVariableOpquantile_3/bias*
_output_shapes
:*
dtype0

quantile_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_3/kernel
x
%quantile_3/kernel/Read/ReadVariableOpReadVariableOpquantile_3/kernel*
_output_shapes
:	р]*
dtype0
v
quantile_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_2/bias
o
#quantile_2/bias/Read/ReadVariableOpReadVariableOpquantile_2/bias*
_output_shapes
:*
dtype0

quantile_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_2/kernel
x
%quantile_2/kernel/Read/ReadVariableOpReadVariableOpquantile_2/kernel*
_output_shapes
:	р]*
dtype0
v
quantile_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_1/bias
o
#quantile_1/bias/Read/ReadVariableOpReadVariableOpquantile_1/bias*
_output_shapes
:*
dtype0

quantile_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_1/kernel
x
%quantile_1/kernel/Read/ReadVariableOpReadVariableOpquantile_1/kernel*
_output_shapes
:	р]*
dtype0
v
quantile_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namequantile_0/bias
o
#quantile_0/bias/Read/ReadVariableOpReadVariableOpquantile_0/bias*
_output_shapes
:*
dtype0

quantile_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	р]*"
shared_namequantile_0/kernel
x
%quantile_0/kernel/Read/ReadVariableOpReadVariableOpquantile_0/kernel*
_output_shapes
:	р]*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:р]*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:р]*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	1р]*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	1р]*
dtype0

embedding_6/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_6/embeddings

*embedding_6/embeddings/Read/ReadVariableOpReadVariableOpembedding_6/embeddings*
_output_shapes

:*
dtype0

embedding_5/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:8*'
shared_nameembedding_5/embeddings

*embedding_5/embeddings/Read/ReadVariableOpReadVariableOpembedding_5/embeddings*
_output_shapes

:8*
dtype0

embedding_4/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_4/embeddings

*embedding_4/embeddings/Read/ReadVariableOpReadVariableOpembedding_4/embeddings*
_output_shapes

:*
dtype0

embedding_3/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_3/embeddings

*embedding_3/embeddings/Read/ReadVariableOpReadVariableOpembedding_3/embeddings*
_output_shapes

:*
dtype0

embedding_2/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_2/embeddings

*embedding_2/embeddings/Read/ReadVariableOpReadVariableOpembedding_2/embeddings*
_output_shapes

:*
dtype0

embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_1/embeddings

*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

:*
dtype0

embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameembedding/embeddings
}
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes

:*
dtype0
z
serving_default_input_1Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_input_2Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_input_3Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_input_4Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_input_5Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_input_6Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_input_7Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
z
serving_default_input_8Placeholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ
Ч	
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2serving_default_input_3serving_default_input_4serving_default_input_5serving_default_input_6serving_default_input_7serving_default_input_8embedding_6/embeddingsembedding_5/embeddingsembedding_4/embeddingsembedding_3/embeddingsembedding_2/embeddingsembedding_1/embeddingsembedding/embeddingsdense/kernel
dense/biasquantile_8/kernelquantile_8/biasquantile_7/kernelquantile_7/biasquantile_6/kernelquantile_6/biasquantile_5/kernelquantile_5/biasquantile_4/kernelquantile_4/biasquantile_3/kernelquantile_3/biasquantile_2/kernelquantile_2/biasquantile_1/kernelquantile_1/biasquantile_0/kernelquantile_0/bias*.
Tin'
%2#*
Tout
2	*
_collective_manager_ids
 *С
_output_shapesЎ
Ћ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*=
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_25808116

NoOpNoOp
у
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ьт
valueСтBНт BЕт
щ
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer_with_weights-7
layer-23
layer-24
layer_with_weights-8
layer-25
layer_with_weights-9
layer-26
layer_with_weights-10
layer-27
layer_with_weights-11
layer-28
layer_with_weights-12
layer-29
layer_with_weights-13
layer-30
 layer_with_weights-14
 layer-31
!layer_with_weights-15
!layer-32
"layer_with_weights-16
"layer-33
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_default_save_signature
*	optimizer
+loss
,
signatures*
* 
* 
* 
* 
* 
* 
* 
 
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3
embeddings*
 
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:
embeddings*
 
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A
embeddings*
 
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H
embeddings*
 
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O
embeddings*
 
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
V
embeddings*
 
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]
embeddings*

^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses* 

d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses* 

j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses* 

p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses* 

v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses* 

|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
* 

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ў
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Ќ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator* 
Ў
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses
Ѓkernel
	Єbias*
Ў
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses
Ћkernel
	Ќbias*
Ў
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses
Гkernel
	Дbias*
Ў
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses
Лkernel
	Мbias*
Ў
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
Уkernel
	Фbias*
Ў
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
Ыkernel
	Ьbias*
Ў
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses
гkernel
	дbias*
Ў
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses
лkernel
	мbias*
Ў
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses
уkernel
	фbias*
ц
30
:1
A2
H3
O4
V5
]6
7
8
Ѓ9
Є10
Ћ11
Ќ12
Г13
Д14
Л15
М16
У17
Ф18
Ы19
Ь20
г21
д22
л23
м24
у25
ф26*
ц
30
:1
A2
H3
O4
V5
]6
7
8
Ѓ9
Є10
Ћ11
Ќ12
Г13
Д14
Л15
М16
У17
Ф18
Ы19
Ь20
г21
д22
л23
м24
у25
ф26*
* 
Е
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
)_default_save_signature
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses*
:
ъtrace_0
ыtrace_1
ьtrace_2
эtrace_3* 
:
юtrace_0
яtrace_1
№trace_2
ёtrace_3* 
* 

ђ
_variables
ѓ_iterations
є_learning_rate
ѕ_index_dict
і
_momentums
ї_velocities
ј_update_step_xla*
* 

љserving_default* 

30*

30*
* 

њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

џtrace_0* 

trace_0* 
hb
VARIABLE_VALUEembedding/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

:0*

:0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses*

trace_0* 

trace_0* 
jd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

A0*

A0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

trace_0* 

trace_0* 
jd
VARIABLE_VALUEembedding_2/embeddings:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

H0*

H0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses*

trace_0* 

trace_0* 
jd
VARIABLE_VALUEembedding_3/embeddings:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

O0*

O0*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

trace_0* 

trace_0* 
jd
VARIABLE_VALUEembedding_4/embeddings:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

V0*

V0*
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses*

Ђtrace_0* 

Ѓtrace_0* 
jd
VARIABLE_VALUEembedding_5/embeddings:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUE*

]0*

]0*
* 

Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses*

Љtrace_0* 

Њtrace_0* 
jd
VARIABLE_VALUEembedding_6/embeddings:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses* 

Аtrace_0* 

Бtrace_0* 
* 
* 
* 

Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses* 

Зtrace_0* 

Иtrace_0* 
* 
* 
* 

Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses* 

Оtrace_0* 

Пtrace_0* 
* 
* 
* 

Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses* 

Хtrace_0* 

Цtrace_0* 
* 
* 
* 

Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses* 

Ьtrace_0* 

Эtrace_0* 
* 
* 
* 

Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

гtrace_0* 

дtrace_0* 
* 
* 
* 

еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

кtrace_0* 

лtrace_0* 
* 
* 
* 

мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

сtrace_0* 

тtrace_0* 

0
1*

0
1*
* 

уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

шtrace_0* 

щtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

яtrace_0
№trace_1* 

ёtrace_0
ђtrace_1* 
* 

Ѓ0
Є1*

Ѓ0
Є1*
* 

ѓnon_trainable_variables
єlayers
ѕmetrics
 іlayer_regularization_losses
їlayer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses*

јtrace_0* 

љtrace_0* 
a[
VARIABLE_VALUEquantile_0/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEquantile_0/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ћ0
Ќ1*

Ћ0
Ќ1*
* 

њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses*

џtrace_0* 

trace_0* 
a[
VARIABLE_VALUEquantile_1/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEquantile_1/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*

Г0
Д1*

Г0
Д1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses*

trace_0* 

trace_0* 
b\
VARIABLE_VALUEquantile_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEquantile_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

Л0
М1*

Л0
М1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses*

trace_0* 

trace_0* 
b\
VARIABLE_VALUEquantile_3/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEquantile_3/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

У0
Ф1*

У0
Ф1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses*

trace_0* 

trace_0* 
b\
VARIABLE_VALUEquantile_4/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEquantile_4/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*

Ы0
Ь1*

Ы0
Ь1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
b\
VARIABLE_VALUEquantile_5/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEquantile_5/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE*

г0
д1*

г0
д1*
* 

non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses*

Ђtrace_0* 

Ѓtrace_0* 
b\
VARIABLE_VALUEquantile_6/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEquantile_6/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE*

л0
м1*

л0
м1*
* 

Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses*

Љtrace_0* 

Њtrace_0* 
b\
VARIABLE_VALUEquantile_7/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEquantile_7/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE*

у0
ф1*

у0
ф1*
* 

Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses*

Аtrace_0* 

Бtrace_0* 
b\
VARIABLE_VALUEquantile_8/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEquantile_8/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33*
T
В0
Г1
Д2
Е3
Ж4
З5
И6
Й7
К8
Л9*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
щ
ѓ0
М1
Н2
О3
П4
Р5
С6
Т7
У8
Ф9
Х10
Ц11
Ч12
Ш13
Щ14
Ъ15
Ы16
Ь17
Э18
Ю19
Я20
а21
б22
в23
г24
д25
е26
ж27
з28
и29
й30
к31
л32
м33
н34
о35
п36
р37
с38
т39
у40
ф41
х42
ц43
ч44
ш45
щ46
ъ47
ы48
ь49
э50
ю51
я52
№53
ё54*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
э
М0
О1
Р2
Т3
Ф4
Ц5
Ш6
Ъ7
Ь8
Ю9
а10
в11
д12
ж13
и14
к15
м16
о17
р18
т19
ф20
ц21
ш22
ъ23
ь24
ю25
№26*
э
Н0
П1
С2
У3
Х4
Ч5
Щ6
Ы7
Э8
Я9
б10
г11
е12
з13
й14
л15
н16
п17
с18
у19
х20
ч21
щ22
ы23
э24
я25
ё26*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
ђ	variables
ѓ	keras_api

єtotal

ѕcount*
<
і	variables
ї	keras_api

јtotal

љcount*
<
њ	variables
ћ	keras_api

ќtotal

§count*
<
ў	variables
џ	keras_api

total

count*
<
	variables
	keras_api

total

count*
<
	variables
	keras_api

total

count*
<
	variables
	keras_api

total

count*
<
	variables
	keras_api

total

count*
<
	variables
	keras_api

total

count*
<
	variables
	keras_api

total

count*
f`
VARIABLE_VALUEAdam/m/embedding/embeddings1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
f`
VARIABLE_VALUEAdam/v/embedding/embeddings1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/embedding_1/embeddings1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/embedding_1/embeddings1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/embedding_2/embeddings1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/embedding_2/embeddings1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/embedding_3/embeddings1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/v/embedding_3/embeddings1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUE*
hb
VARIABLE_VALUEAdam/m/embedding_4/embeddings1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/embedding_4/embeddings2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/embedding_5/embeddings2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/embedding_5/embeddings2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/m/embedding_6/embeddings2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEAdam/v/embedding_6/embeddings2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/m/dense/kernel2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEAdam/v/dense/kernel2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/m/dense/bias2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEAdam/v/dense/bias2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_0/kernel2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_0/kernel2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_0/bias2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_0/bias2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_1/kernel2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_1/kernel2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_1/bias2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_1/bias2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_2/kernel2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_2/kernel2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_2/bias2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_2/bias2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_3/kernel2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_3/kernel2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_3/bias2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_3/bias2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_4/kernel2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_4/kernel2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_4/bias2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_4/bias2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_5/kernel2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_5/kernel2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_5/bias2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_5/bias2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_6/kernel2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_6/kernel2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_6/bias2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_6/bias2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_7/kernel2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_7/kernel2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_7/bias2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_7/bias2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/m/quantile_8/kernel2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUEAdam/v/quantile_8/kernel2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/m/quantile_8/bias2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUEAdam/v/quantile_8/bias2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUE*

є0
ѕ1*

ђ	variables*
UO
VARIABLE_VALUEtotal_94keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_94keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

ј0
љ1*

і	variables*
UO
VARIABLE_VALUEtotal_84keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_84keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*

ќ0
§1*

њ	variables*
UO
VARIABLE_VALUEtotal_74keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_74keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

ў	variables*
UO
VARIABLE_VALUEtotal_64keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_64keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_54keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_54keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_44keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_44keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_34keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_34keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_24keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_24keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
д
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsembedding_3/embeddingsembedding_4/embeddingsembedding_5/embeddingsembedding_6/embeddingsdense/kernel
dense/biasquantile_0/kernelquantile_0/biasquantile_1/kernelquantile_1/biasquantile_2/kernelquantile_2/biasquantile_3/kernelquantile_3/biasquantile_4/kernelquantile_4/biasquantile_5/kernelquantile_5/biasquantile_6/kernelquantile_6/biasquantile_7/kernelquantile_7/biasquantile_8/kernelquantile_8/bias	iterationlearning_rateAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/embedding_1/embeddingsAdam/v/embedding_1/embeddingsAdam/m/embedding_2/embeddingsAdam/v/embedding_2/embeddingsAdam/m/embedding_3/embeddingsAdam/v/embedding_3/embeddingsAdam/m/embedding_4/embeddingsAdam/v/embedding_4/embeddingsAdam/m/embedding_5/embeddingsAdam/v/embedding_5/embeddingsAdam/m/embedding_6/embeddingsAdam/v/embedding_6/embeddingsAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/quantile_0/kernelAdam/v/quantile_0/kernelAdam/m/quantile_0/biasAdam/v/quantile_0/biasAdam/m/quantile_1/kernelAdam/v/quantile_1/kernelAdam/m/quantile_1/biasAdam/v/quantile_1/biasAdam/m/quantile_2/kernelAdam/v/quantile_2/kernelAdam/m/quantile_2/biasAdam/v/quantile_2/biasAdam/m/quantile_3/kernelAdam/v/quantile_3/kernelAdam/m/quantile_3/biasAdam/v/quantile_3/biasAdam/m/quantile_4/kernelAdam/v/quantile_4/kernelAdam/m/quantile_4/biasAdam/v/quantile_4/biasAdam/m/quantile_5/kernelAdam/v/quantile_5/kernelAdam/m/quantile_5/biasAdam/v/quantile_5/biasAdam/m/quantile_6/kernelAdam/v/quantile_6/kernelAdam/m/quantile_6/biasAdam/v/quantile_6/biasAdam/m/quantile_7/kernelAdam/v/quantile_7/kernelAdam/m/quantile_7/biasAdam/v/quantile_7/biasAdam/m/quantile_8/kernelAdam/v/quantile_8/kernelAdam/m/quantile_8/biasAdam/v/quantile_8/biastotal_9count_9total_8count_8total_7count_7total_6count_6total_5count_5total_4count_4total_3count_3total_2count_2total_1count_1totalcountConst*t
Tinm
k2i*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_25809660
Я
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameembedding/embeddingsembedding_1/embeddingsembedding_2/embeddingsembedding_3/embeddingsembedding_4/embeddingsembedding_5/embeddingsembedding_6/embeddingsdense/kernel
dense/biasquantile_0/kernelquantile_0/biasquantile_1/kernelquantile_1/biasquantile_2/kernelquantile_2/biasquantile_3/kernelquantile_3/biasquantile_4/kernelquantile_4/biasquantile_5/kernelquantile_5/biasquantile_6/kernelquantile_6/biasquantile_7/kernelquantile_7/biasquantile_8/kernelquantile_8/bias	iterationlearning_rateAdam/m/embedding/embeddingsAdam/v/embedding/embeddingsAdam/m/embedding_1/embeddingsAdam/v/embedding_1/embeddingsAdam/m/embedding_2/embeddingsAdam/v/embedding_2/embeddingsAdam/m/embedding_3/embeddingsAdam/v/embedding_3/embeddingsAdam/m/embedding_4/embeddingsAdam/v/embedding_4/embeddingsAdam/m/embedding_5/embeddingsAdam/v/embedding_5/embeddingsAdam/m/embedding_6/embeddingsAdam/v/embedding_6/embeddingsAdam/m/dense/kernelAdam/v/dense/kernelAdam/m/dense/biasAdam/v/dense/biasAdam/m/quantile_0/kernelAdam/v/quantile_0/kernelAdam/m/quantile_0/biasAdam/v/quantile_0/biasAdam/m/quantile_1/kernelAdam/v/quantile_1/kernelAdam/m/quantile_1/biasAdam/v/quantile_1/biasAdam/m/quantile_2/kernelAdam/v/quantile_2/kernelAdam/m/quantile_2/biasAdam/v/quantile_2/biasAdam/m/quantile_3/kernelAdam/v/quantile_3/kernelAdam/m/quantile_3/biasAdam/v/quantile_3/biasAdam/m/quantile_4/kernelAdam/v/quantile_4/kernelAdam/m/quantile_4/biasAdam/v/quantile_4/biasAdam/m/quantile_5/kernelAdam/v/quantile_5/kernelAdam/m/quantile_5/biasAdam/v/quantile_5/biasAdam/m/quantile_6/kernelAdam/v/quantile_6/kernelAdam/m/quantile_6/biasAdam/v/quantile_6/biasAdam/m/quantile_7/kernelAdam/v/quantile_7/kernelAdam/m/quantile_7/biasAdam/v/quantile_7/biasAdam/m/quantile_8/kernelAdam/v/quantile_8/kernelAdam/m/quantile_8/biasAdam/v/quantile_8/biastotal_9count_9total_8count_8total_7count_7total_6count_6total_5count_5total_4count_4total_3count_3total_2count_2total_1count_1totalcount*s
Tinl
j2h*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_25809979уЅ
Ќ

d
E__inference_dropout_layer_call_and_return_conditional_losses_25807022

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџр]:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
м
c
E__inference_dropout_layer_call_and_return_conditional_losses_25807225

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџр]\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџр]:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
ђЃ

C__inference_model_layer_call_and_return_conditional_losses_25808565
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_77
%embedding_6_embedding_lookup_25808437:7
%embedding_5_embedding_lookup_25808443:87
%embedding_4_embedding_lookup_25808449:7
%embedding_3_embedding_lookup_25808455:7
%embedding_2_embedding_lookup_25808461:7
%embedding_1_embedding_lookup_25808467:5
#embedding_embedding_lookup_25808473:7
$dense_matmul_readvariableop_resource:	1р]4
%dense_biasadd_readvariableop_resource:	р]<
)quantile_8_matmul_readvariableop_resource:	р]8
*quantile_8_biasadd_readvariableop_resource:<
)quantile_7_matmul_readvariableop_resource:	р]8
*quantile_7_biasadd_readvariableop_resource:<
)quantile_6_matmul_readvariableop_resource:	р]8
*quantile_6_biasadd_readvariableop_resource:<
)quantile_5_matmul_readvariableop_resource:	р]8
*quantile_5_biasadd_readvariableop_resource:<
)quantile_4_matmul_readvariableop_resource:	р]8
*quantile_4_biasadd_readvariableop_resource:<
)quantile_3_matmul_readvariableop_resource:	р]8
*quantile_3_biasadd_readvariableop_resource:<
)quantile_2_matmul_readvariableop_resource:	р]8
*quantile_2_biasadd_readvariableop_resource:<
)quantile_1_matmul_readvariableop_resource:	р]8
*quantile_1_biasadd_readvariableop_resource:<
)quantile_0_matmul_readvariableop_resource:	р]8
*quantile_0_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8Ђdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂembedding/embedding_lookupЂembedding_1/embedding_lookupЂembedding_2/embedding_lookupЂembedding_3/embedding_lookupЂembedding_4/embedding_lookupЂembedding_5/embedding_lookupЂembedding_6/embedding_lookupЂ!quantile_0/BiasAdd/ReadVariableOpЂ quantile_0/MatMul/ReadVariableOpЂ!quantile_1/BiasAdd/ReadVariableOpЂ quantile_1/MatMul/ReadVariableOpЂ!quantile_2/BiasAdd/ReadVariableOpЂ quantile_2/MatMul/ReadVariableOpЂ!quantile_3/BiasAdd/ReadVariableOpЂ quantile_3/MatMul/ReadVariableOpЂ!quantile_4/BiasAdd/ReadVariableOpЂ quantile_4/MatMul/ReadVariableOpЂ!quantile_5/BiasAdd/ReadVariableOpЂ quantile_5/MatMul/ReadVariableOpЂ!quantile_6/BiasAdd/ReadVariableOpЂ quantile_6/MatMul/ReadVariableOpЂ!quantile_7/BiasAdd/ReadVariableOpЂ quantile_7/MatMul/ReadVariableOpЂ!quantile_8/BiasAdd/ReadVariableOpЂ quantile_8/MatMul/ReadVariableOpc
embedding_6/CastCastinputs_6*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_6/embedding_lookupResourceGather%embedding_6_embedding_lookup_25808437embedding_6/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_6/embedding_lookup/25808437*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_6/embedding_lookup/25808437*+
_output_shapes
:џџџџџџџџџ
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_5/CastCastinputs_5*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_5/embedding_lookupResourceGather%embedding_5_embedding_lookup_25808443embedding_5/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_5/embedding_lookup/25808443*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_5/embedding_lookup/25808443*+
_output_shapes
:џџџџџџџџџ
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_4/CastCastinputs_4*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_4/embedding_lookupResourceGather%embedding_4_embedding_lookup_25808449embedding_4/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_4/embedding_lookup/25808449*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_4/embedding_lookup/25808449*+
_output_shapes
:џџџџџџџџџ
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_3/CastCastinputs_3*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_3/embedding_lookupResourceGather%embedding_3_embedding_lookup_25808455embedding_3/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_3/embedding_lookup/25808455*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_3/embedding_lookup/25808455*+
_output_shapes
:џџџџџџџџџ
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_2/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_2/embedding_lookupResourceGather%embedding_2_embedding_lookup_25808461embedding_2/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_2/embedding_lookup/25808461*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_2/embedding_lookup/25808461*+
_output_shapes
:џџџџџџџџџ
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_1/embedding_lookupResourceGather%embedding_1_embedding_lookup_25808467embedding_1/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_1/embedding_lookup/25808467*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_1/embedding_lookup/25808467*+
_output_shapes
:џџџџџџџџџ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџa
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџч
embedding/embedding_lookupResourceGather#embedding_embedding_lookup_25808473embedding/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding/embedding_lookup/25808473*+
_output_shapes
:џџџџџџџџџ*
dtype0Т
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding/embedding_lookup/25808473*+
_output_shapes
:џџџџџџџџџ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten/ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_1/ReshapeReshape0embedding_1/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_2/ReshapeReshape0embedding_2/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_3/ReshapeReshape0embedding_3/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_4/ReshapeReshape0embedding_4/embedding_lookup/Identity_1:output:0flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_5/ReshapeReshape0embedding_5/embedding_lookup/Identity_1:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_6/ReshapeReshape0embedding_6/embedding_lookup/Identity_1:output:0flatten_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ч
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0flatten_5/Reshape:output:0flatten_6/Reshape:output:0inputs_7 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ1
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	1р]*
dtype0
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:р]*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]i
dropout/IdentityIdentitydense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџр]
 quantile_8/MatMul/ReadVariableOpReadVariableOp)quantile_8_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_8/MatMulMatMuldropout/Identity:output:0(quantile_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_8/BiasAdd/ReadVariableOpReadVariableOp*quantile_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_8/BiasAddBiasAddquantile_8/MatMul:product:0)quantile_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_7/MatMul/ReadVariableOpReadVariableOp)quantile_7_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_7/MatMulMatMuldropout/Identity:output:0(quantile_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_7/BiasAdd/ReadVariableOpReadVariableOp*quantile_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_7/BiasAddBiasAddquantile_7/MatMul:product:0)quantile_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_6/MatMul/ReadVariableOpReadVariableOp)quantile_6_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_6/MatMulMatMuldropout/Identity:output:0(quantile_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_6/BiasAdd/ReadVariableOpReadVariableOp*quantile_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_6/BiasAddBiasAddquantile_6/MatMul:product:0)quantile_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_5/MatMul/ReadVariableOpReadVariableOp)quantile_5_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_5/MatMulMatMuldropout/Identity:output:0(quantile_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_5/BiasAdd/ReadVariableOpReadVariableOp*quantile_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_5/BiasAddBiasAddquantile_5/MatMul:product:0)quantile_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_4/MatMul/ReadVariableOpReadVariableOp)quantile_4_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_4/MatMulMatMuldropout/Identity:output:0(quantile_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_4/BiasAdd/ReadVariableOpReadVariableOp*quantile_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_4/BiasAddBiasAddquantile_4/MatMul:product:0)quantile_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_3/MatMul/ReadVariableOpReadVariableOp)quantile_3_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_3/MatMulMatMuldropout/Identity:output:0(quantile_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_3/BiasAdd/ReadVariableOpReadVariableOp*quantile_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_3/BiasAddBiasAddquantile_3/MatMul:product:0)quantile_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_2/MatMul/ReadVariableOpReadVariableOp)quantile_2_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_2/MatMulMatMuldropout/Identity:output:0(quantile_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_2/BiasAdd/ReadVariableOpReadVariableOp*quantile_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_2/BiasAddBiasAddquantile_2/MatMul:product:0)quantile_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_1/MatMul/ReadVariableOpReadVariableOp)quantile_1_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_1/MatMulMatMuldropout/Identity:output:0(quantile_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_1/BiasAdd/ReadVariableOpReadVariableOp*quantile_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_1/BiasAddBiasAddquantile_1/MatMul:product:0)quantile_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_0/MatMul/ReadVariableOpReadVariableOp)quantile_0_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_0/MatMulMatMuldropout/Identity:output:0(quantile_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_0/BiasAdd/ReadVariableOpReadVariableOp*quantile_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_0/BiasAddBiasAddquantile_0/MatMul:product:0)quantile_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentityquantile_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_1Identityquantile_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_2Identityquantile_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_3Identityquantile_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_4Identityquantile_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_5Identityquantile_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_6Identityquantile_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_7Identityquantile_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_8Identityquantile_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџй
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup^embedding_1/embedding_lookup^embedding_2/embedding_lookup^embedding_3/embedding_lookup^embedding_4/embedding_lookup^embedding_5/embedding_lookup^embedding_6/embedding_lookup"^quantile_0/BiasAdd/ReadVariableOp!^quantile_0/MatMul/ReadVariableOp"^quantile_1/BiasAdd/ReadVariableOp!^quantile_1/MatMul/ReadVariableOp"^quantile_2/BiasAdd/ReadVariableOp!^quantile_2/MatMul/ReadVariableOp"^quantile_3/BiasAdd/ReadVariableOp!^quantile_3/MatMul/ReadVariableOp"^quantile_4/BiasAdd/ReadVariableOp!^quantile_4/MatMul/ReadVariableOp"^quantile_5/BiasAdd/ReadVariableOp!^quantile_5/MatMul/ReadVariableOp"^quantile_6/BiasAdd/ReadVariableOp!^quantile_6/MatMul/ReadVariableOp"^quantile_7/BiasAdd/ReadVariableOp!^quantile_7/MatMul/ReadVariableOp"^quantile_8/BiasAdd/ReadVariableOp!^quantile_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2F
!quantile_0/BiasAdd/ReadVariableOp!quantile_0/BiasAdd/ReadVariableOp2D
 quantile_0/MatMul/ReadVariableOp quantile_0/MatMul/ReadVariableOp2F
!quantile_1/BiasAdd/ReadVariableOp!quantile_1/BiasAdd/ReadVariableOp2D
 quantile_1/MatMul/ReadVariableOp quantile_1/MatMul/ReadVariableOp2F
!quantile_2/BiasAdd/ReadVariableOp!quantile_2/BiasAdd/ReadVariableOp2D
 quantile_2/MatMul/ReadVariableOp quantile_2/MatMul/ReadVariableOp2F
!quantile_3/BiasAdd/ReadVariableOp!quantile_3/BiasAdd/ReadVariableOp2D
 quantile_3/MatMul/ReadVariableOp quantile_3/MatMul/ReadVariableOp2F
!quantile_4/BiasAdd/ReadVariableOp!quantile_4/BiasAdd/ReadVariableOp2D
 quantile_4/MatMul/ReadVariableOp quantile_4/MatMul/ReadVariableOp2F
!quantile_5/BiasAdd/ReadVariableOp!quantile_5/BiasAdd/ReadVariableOp2D
 quantile_5/MatMul/ReadVariableOp quantile_5/MatMul/ReadVariableOp2F
!quantile_6/BiasAdd/ReadVariableOp!quantile_6/BiasAdd/ReadVariableOp2D
 quantile_6/MatMul/ReadVariableOp quantile_6/MatMul/ReadVariableOp2F
!quantile_7/BiasAdd/ReadVariableOp!quantile_7/BiasAdd/ReadVariableOp2D
 quantile_7/MatMul/ReadVariableOp quantile_7/MatMul/ReadVariableOp2F
!quantile_8/BiasAdd/ReadVariableOp!quantile_8/BiasAdd/ReadVariableOp2D
 quantile_8/MatMul/ReadVariableOp quantile_8/MatMul/ReadVariableOp:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Ћ
H
,__inference_flatten_6_layer_call_fn_25808755

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_25806976`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_3_layer_call_and_return_conditional_losses_25807114

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Э

-__inference_quantile_1_layer_call_fn_25808861

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_1_layer_call_and_return_conditional_losses_25807146o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Ћ
H
,__inference_flatten_1_layer_call_fn_25808700

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_25806936`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_2_layer_call_and_return_conditional_losses_25807130

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_2_layer_call_and_return_conditional_losses_25808890

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Ќ

.__inference_embedding_4_layer_call_fn_25808640

inputs
unknown:
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_4_layer_call_and_return_conditional_losses_25806862s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ

.__inference_embedding_6_layer_call_fn_25808674

inputs
unknown:
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_6_layer_call_and_return_conditional_losses_25806834s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ	
І
G__inference_embedding_layer_call_and_return_conditional_losses_25806918

inputs+
embedding_lookup_25806912:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25806912Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25806912*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25806912*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э

-__inference_quantile_8_layer_call_fn_25808994

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_8_layer_call_and_return_conditional_losses_25807034o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Ѓ
F
*__inference_dropout_layer_call_fn_25808816

inputs
identityБ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_25807225a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџр]:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
П
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_25808750

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_1_layer_call_and_return_conditional_losses_25808599

inputs+
embedding_lookup_25808593:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25808593Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25808593*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25808593*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ

.__inference_embedding_2_layer_call_fn_25808606

inputs
unknown:
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_25806890s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_2_layer_call_and_return_conditional_losses_25808616

inputs+
embedding_lookup_25808610:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25808610Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25808610*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25808610*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ё
а
C__inference_model_layer_call_and_return_conditional_losses_25807177
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8&
embedding_6_25806835:&
embedding_5_25806849:8&
embedding_4_25806863:&
embedding_3_25806877:&
embedding_2_25806891:&
embedding_1_25806905:$
embedding_25806919:!
dense_25807005:	1р]
dense_25807007:	р]&
quantile_8_25807035:	р]!
quantile_8_25807037:&
quantile_7_25807051:	р]!
quantile_7_25807053:&
quantile_6_25807067:	р]!
quantile_6_25807069:&
quantile_5_25807083:	р]!
quantile_5_25807085:&
quantile_4_25807099:	р]!
quantile_4_25807101:&
quantile_3_25807115:	р]!
quantile_3_25807117:&
quantile_2_25807131:	р]!
quantile_2_25807133:&
quantile_1_25807147:	р]!
quantile_1_25807149:&
quantile_0_25807163:	р]!
quantile_0_25807165:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8Ђdense/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂ#embedding_1/StatefulPartitionedCallЂ#embedding_2/StatefulPartitionedCallЂ#embedding_3/StatefulPartitionedCallЂ#embedding_4/StatefulPartitionedCallЂ#embedding_5/StatefulPartitionedCallЂ#embedding_6/StatefulPartitionedCallЂ"quantile_0/StatefulPartitionedCallЂ"quantile_1/StatefulPartitionedCallЂ"quantile_2/StatefulPartitionedCallЂ"quantile_3/StatefulPartitionedCallЂ"quantile_4/StatefulPartitionedCallЂ"quantile_5/StatefulPartitionedCallЂ"quantile_6/StatefulPartitionedCallЂ"quantile_7/StatefulPartitionedCallЂ"quantile_8/StatefulPartitionedCallя
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinput_7embedding_6_25806835*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_6_layer_call_and_return_conditional_losses_25806834я
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallinput_6embedding_5_25806849*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_5_layer_call_and_return_conditional_losses_25806848я
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallinput_5embedding_4_25806863*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_4_layer_call_and_return_conditional_losses_25806862я
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_4embedding_3_25806877*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_3_layer_call_and_return_conditional_losses_25806876я
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3embedding_2_25806891*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_25806890я
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_25806905*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_25806904щ
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_25806919*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_25806918м
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_25806928т
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_25806936т
flatten_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_25806944т
flatten_3/PartitionedCallPartitionedCall,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_25806952т
flatten_4/PartitionedCallPartitionedCall,embedding_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_25806960т
flatten_5/PartitionedCallPartitionedCall,embedding_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_25806968т
flatten_6/PartitionedCallPartitionedCall,embedding_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_25806976Т
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0input_8*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_25806991
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_25807005dense_25807007*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_25807004щ
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_25807022 
"quantile_8/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_8_25807035quantile_8_25807037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_8_layer_call_and_return_conditional_losses_25807034 
"quantile_7/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_7_25807051quantile_7_25807053*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_7_layer_call_and_return_conditional_losses_25807050 
"quantile_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_6_25807067quantile_6_25807069*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_6_layer_call_and_return_conditional_losses_25807066 
"quantile_5/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_5_25807083quantile_5_25807085*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_5_layer_call_and_return_conditional_losses_25807082 
"quantile_4/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_4_25807099quantile_4_25807101*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_4_layer_call_and_return_conditional_losses_25807098 
"quantile_3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_3_25807115quantile_3_25807117*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_3_layer_call_and_return_conditional_losses_25807114 
"quantile_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_2_25807131quantile_2_25807133*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_2_layer_call_and_return_conditional_losses_25807130 
"quantile_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_1_25807147quantile_1_25807149*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_1_layer_call_and_return_conditional_losses_25807146 
"quantile_0/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_0_25807163quantile_0_25807165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_0_layer_call_and_return_conditional_losses_25807162z
IdentityIdentity+quantile_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_1Identity+quantile_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_2Identity+quantile_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_3Identity+quantile_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_4Identity+quantile_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_5Identity+quantile_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_6Identity+quantile_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_7Identity+quantile_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_8Identity+quantile_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџн
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall#^quantile_0/StatefulPartitionedCall#^quantile_1/StatefulPartitionedCall#^quantile_2/StatefulPartitionedCall#^quantile_3/StatefulPartitionedCall#^quantile_4/StatefulPartitionedCall#^quantile_5/StatefulPartitionedCall#^quantile_6/StatefulPartitionedCall#^quantile_7/StatefulPartitionedCall#^quantile_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2H
"quantile_0/StatefulPartitionedCall"quantile_0/StatefulPartitionedCall2H
"quantile_1/StatefulPartitionedCall"quantile_1/StatefulPartitionedCall2H
"quantile_2/StatefulPartitionedCall"quantile_2/StatefulPartitionedCall2H
"quantile_3/StatefulPartitionedCall"quantile_3/StatefulPartitionedCall2H
"quantile_4/StatefulPartitionedCall"quantile_4/StatefulPartitionedCall2H
"quantile_5/StatefulPartitionedCall"quantile_5/StatefulPartitionedCall2H
"quantile_6/StatefulPartitionedCall"quantile_6/StatefulPartitionedCall2H
"quantile_7/StatefulPartitionedCall"quantile_7/StatefulPartitionedCall2H
"quantile_8/StatefulPartitionedCall"quantile_8/StatefulPartitionedCall:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
ѕ
c
*__inference_dropout_layer_call_fn_25808811

inputs
identityЂStatefulPartitionedCallС
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_25807022p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџр]`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџр]22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
а"
Н
(__inference_model_layer_call_fn_25807644
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8
unknown:
	unknown_0:8
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:	1р]
	unknown_7:	р]
	unknown_8:	р]
	unknown_9:

unknown_10:	р]

unknown_11:

unknown_12:	р]

unknown_13:

unknown_14:	р]

unknown_15:

unknown_16:	р]

unknown_17:

unknown_18:	р]

unknown_19:

unknown_20:	р]

unknown_21:

unknown_22:	р]

unknown_23:

unknown_24:	р]

unknown_25:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*.
Tin'
%2#*
Tout
2	*
_collective_manager_ids
 *С
_output_shapesЎ
Ћ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*=
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_25807571o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
П
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_25808728

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
м
c
E__inference_dropout_layer_call_and_return_conditional_losses_25808833

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:џџџџџџџџџр]\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџр]:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs

Д
C__inference_model_layer_call_and_return_conditional_losses_25807571

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7&
embedding_6_25807482:&
embedding_5_25807485:8&
embedding_4_25807488:&
embedding_3_25807491:&
embedding_2_25807494:&
embedding_1_25807497:$
embedding_25807500:!
dense_25807511:	1р]
dense_25807513:	р]&
quantile_8_25807517:	р]!
quantile_8_25807519:&
quantile_7_25807522:	р]!
quantile_7_25807524:&
quantile_6_25807527:	р]!
quantile_6_25807529:&
quantile_5_25807532:	р]!
quantile_5_25807534:&
quantile_4_25807537:	р]!
quantile_4_25807539:&
quantile_3_25807542:	р]!
quantile_3_25807544:&
quantile_2_25807547:	р]!
quantile_2_25807549:&
quantile_1_25807552:	р]!
quantile_1_25807554:&
quantile_0_25807557:	р]!
quantile_0_25807559:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8Ђdense/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂ#embedding_1/StatefulPartitionedCallЂ#embedding_2/StatefulPartitionedCallЂ#embedding_3/StatefulPartitionedCallЂ#embedding_4/StatefulPartitionedCallЂ#embedding_5/StatefulPartitionedCallЂ#embedding_6/StatefulPartitionedCallЂ"quantile_0/StatefulPartitionedCallЂ"quantile_1/StatefulPartitionedCallЂ"quantile_2/StatefulPartitionedCallЂ"quantile_3/StatefulPartitionedCallЂ"quantile_4/StatefulPartitionedCallЂ"quantile_5/StatefulPartitionedCallЂ"quantile_6/StatefulPartitionedCallЂ"quantile_7/StatefulPartitionedCallЂ"quantile_8/StatefulPartitionedCall№
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinputs_6embedding_6_25807482*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_6_layer_call_and_return_conditional_losses_25806834№
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallinputs_5embedding_5_25807485*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_5_layer_call_and_return_conditional_losses_25806848№
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputs_4embedding_4_25807488*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_4_layer_call_and_return_conditional_losses_25806862№
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_3_25807491*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_3_layer_call_and_return_conditional_losses_25806876№
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_2_25807494*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_25806890№
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_1_25807497*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_25806904ш
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_25807500*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_25806918м
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_25806928т
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_25806936т
flatten_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_25806944т
flatten_3/PartitionedCallPartitionedCall,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_25806952т
flatten_4/PartitionedCallPartitionedCall,embedding_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_25806960т
flatten_5/PartitionedCallPartitionedCall,embedding_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_25806968т
flatten_6/PartitionedCallPartitionedCall,embedding_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_25806976У
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0inputs_7*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_25806991
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_25807511dense_25807513*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_25807004й
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_25807225
"quantile_8/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_8_25807517quantile_8_25807519*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_8_layer_call_and_return_conditional_losses_25807034
"quantile_7/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_7_25807522quantile_7_25807524*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_7_layer_call_and_return_conditional_losses_25807050
"quantile_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_6_25807527quantile_6_25807529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_6_layer_call_and_return_conditional_losses_25807066
"quantile_5/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_5_25807532quantile_5_25807534*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_5_layer_call_and_return_conditional_losses_25807082
"quantile_4/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_4_25807537quantile_4_25807539*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_4_layer_call_and_return_conditional_losses_25807098
"quantile_3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_3_25807542quantile_3_25807544*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_3_layer_call_and_return_conditional_losses_25807114
"quantile_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_2_25807547quantile_2_25807549*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_2_layer_call_and_return_conditional_losses_25807130
"quantile_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_1_25807552quantile_1_25807554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_1_layer_call_and_return_conditional_losses_25807146
"quantile_0/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_0_25807557quantile_0_25807559*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_0_layer_call_and_return_conditional_losses_25807162z
IdentityIdentity+quantile_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_1Identity+quantile_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_2Identity+quantile_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_3Identity+quantile_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_4Identity+quantile_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_5Identity+quantile_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_6Identity+quantile_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_7Identity+quantile_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_8Identity+quantile_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЛ
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall#^quantile_0/StatefulPartitionedCall#^quantile_1/StatefulPartitionedCall#^quantile_2/StatefulPartitionedCall#^quantile_3/StatefulPartitionedCall#^quantile_4/StatefulPartitionedCall#^quantile_5/StatefulPartitionedCall#^quantile_6/StatefulPartitionedCall#^quantile_7/StatefulPartitionedCall#^quantile_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2H
"quantile_0/StatefulPartitionedCall"quantile_0/StatefulPartitionedCall2H
"quantile_1/StatefulPartitionedCall"quantile_1/StatefulPartitionedCall2H
"quantile_2/StatefulPartitionedCall"quantile_2/StatefulPartitionedCall2H
"quantile_3/StatefulPartitionedCall"quantile_3/StatefulPartitionedCall2H
"quantile_4/StatefulPartitionedCall"quantile_4/StatefulPartitionedCall2H
"quantile_5/StatefulPartitionedCall"quantile_5/StatefulPartitionedCall2H
"quantile_6/StatefulPartitionedCall"quantile_6/StatefulPartitionedCall2H
"quantile_7/StatefulPartitionedCall"quantile_7/StatefulPartitionedCall2H
"quantile_8/StatefulPartitionedCall"quantile_8/StatefulPartitionedCall:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ў"
Л
&__inference_signature_wrapper_25808116
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8
unknown:
	unknown_0:8
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:	1р]
	unknown_7:	р]
	unknown_8:	р]
	unknown_9:

unknown_10:	р]

unknown_11:

unknown_12:	р]

unknown_13:

unknown_14:	р]

unknown_15:

unknown_16:	р]

unknown_17:

unknown_18:	р]

unknown_19:

unknown_20:	р]

unknown_21:

unknown_22:	р]

unknown_23:

unknown_24:	р]

unknown_25:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8ЂStatefulPartitionedCallї
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*.
Tin'
%2#*
Tout
2	*
_collective_manager_ids
 *С
_output_shapesЎ
Ћ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*=
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_25806813o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
П
c
G__inference_flatten_5_layer_call_and_return_conditional_losses_25806968

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_1_layer_call_and_return_conditional_losses_25807146

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_8_layer_call_and_return_conditional_losses_25807034

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Д
Љ
#__inference__wrapped_model_25806813
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8=
+model_embedding_6_embedding_lookup_25806685:=
+model_embedding_5_embedding_lookup_25806691:8=
+model_embedding_4_embedding_lookup_25806697:=
+model_embedding_3_embedding_lookup_25806703:=
+model_embedding_2_embedding_lookup_25806709:=
+model_embedding_1_embedding_lookup_25806715:;
)model_embedding_embedding_lookup_25806721:=
*model_dense_matmul_readvariableop_resource:	1р]:
+model_dense_biasadd_readvariableop_resource:	р]B
/model_quantile_8_matmul_readvariableop_resource:	р]>
0model_quantile_8_biasadd_readvariableop_resource:B
/model_quantile_7_matmul_readvariableop_resource:	р]>
0model_quantile_7_biasadd_readvariableop_resource:B
/model_quantile_6_matmul_readvariableop_resource:	р]>
0model_quantile_6_biasadd_readvariableop_resource:B
/model_quantile_5_matmul_readvariableop_resource:	р]>
0model_quantile_5_biasadd_readvariableop_resource:B
/model_quantile_4_matmul_readvariableop_resource:	р]>
0model_quantile_4_biasadd_readvariableop_resource:B
/model_quantile_3_matmul_readvariableop_resource:	р]>
0model_quantile_3_biasadd_readvariableop_resource:B
/model_quantile_2_matmul_readvariableop_resource:	р]>
0model_quantile_2_biasadd_readvariableop_resource:B
/model_quantile_1_matmul_readvariableop_resource:	р]>
0model_quantile_1_biasadd_readvariableop_resource:B
/model_quantile_0_matmul_readvariableop_resource:	р]>
0model_quantile_0_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8Ђ"model/dense/BiasAdd/ReadVariableOpЂ!model/dense/MatMul/ReadVariableOpЂ model/embedding/embedding_lookupЂ"model/embedding_1/embedding_lookupЂ"model/embedding_2/embedding_lookupЂ"model/embedding_3/embedding_lookupЂ"model/embedding_4/embedding_lookupЂ"model/embedding_5/embedding_lookupЂ"model/embedding_6/embedding_lookupЂ'model/quantile_0/BiasAdd/ReadVariableOpЂ&model/quantile_0/MatMul/ReadVariableOpЂ'model/quantile_1/BiasAdd/ReadVariableOpЂ&model/quantile_1/MatMul/ReadVariableOpЂ'model/quantile_2/BiasAdd/ReadVariableOpЂ&model/quantile_2/MatMul/ReadVariableOpЂ'model/quantile_3/BiasAdd/ReadVariableOpЂ&model/quantile_3/MatMul/ReadVariableOpЂ'model/quantile_4/BiasAdd/ReadVariableOpЂ&model/quantile_4/MatMul/ReadVariableOpЂ'model/quantile_5/BiasAdd/ReadVariableOpЂ&model/quantile_5/MatMul/ReadVariableOpЂ'model/quantile_6/BiasAdd/ReadVariableOpЂ&model/quantile_6/MatMul/ReadVariableOpЂ'model/quantile_7/BiasAdd/ReadVariableOpЂ&model/quantile_7/MatMul/ReadVariableOpЂ'model/quantile_8/BiasAdd/ReadVariableOpЂ&model/quantile_8/MatMul/ReadVariableOph
model/embedding_6/CastCastinput_7*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
"model/embedding_6/embedding_lookupResourceGather+model_embedding_6_embedding_lookup_25806685model/embedding_6/Cast:y:0*
Tindices0*>
_class4
20loc:@model/embedding_6/embedding_lookup/25806685*+
_output_shapes
:џџџџџџџџџ*
dtype0к
+model/embedding_6/embedding_lookup/IdentityIdentity+model/embedding_6/embedding_lookup:output:0*
T0*>
_class4
20loc:@model/embedding_6/embedding_lookup/25806685*+
_output_shapes
:џџџџџџџџџЅ
-model/embedding_6/embedding_lookup/Identity_1Identity4model/embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџh
model/embedding_5/CastCastinput_6*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
"model/embedding_5/embedding_lookupResourceGather+model_embedding_5_embedding_lookup_25806691model/embedding_5/Cast:y:0*
Tindices0*>
_class4
20loc:@model/embedding_5/embedding_lookup/25806691*+
_output_shapes
:џџџџџџџџџ*
dtype0к
+model/embedding_5/embedding_lookup/IdentityIdentity+model/embedding_5/embedding_lookup:output:0*
T0*>
_class4
20loc:@model/embedding_5/embedding_lookup/25806691*+
_output_shapes
:џџџџџџџџџЅ
-model/embedding_5/embedding_lookup/Identity_1Identity4model/embedding_5/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџh
model/embedding_4/CastCastinput_5*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
"model/embedding_4/embedding_lookupResourceGather+model_embedding_4_embedding_lookup_25806697model/embedding_4/Cast:y:0*
Tindices0*>
_class4
20loc:@model/embedding_4/embedding_lookup/25806697*+
_output_shapes
:џџџџџџџџџ*
dtype0к
+model/embedding_4/embedding_lookup/IdentityIdentity+model/embedding_4/embedding_lookup:output:0*
T0*>
_class4
20loc:@model/embedding_4/embedding_lookup/25806697*+
_output_shapes
:џџџџџџџџџЅ
-model/embedding_4/embedding_lookup/Identity_1Identity4model/embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџh
model/embedding_3/CastCastinput_4*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
"model/embedding_3/embedding_lookupResourceGather+model_embedding_3_embedding_lookup_25806703model/embedding_3/Cast:y:0*
Tindices0*>
_class4
20loc:@model/embedding_3/embedding_lookup/25806703*+
_output_shapes
:џџџџџџџџџ*
dtype0к
+model/embedding_3/embedding_lookup/IdentityIdentity+model/embedding_3/embedding_lookup:output:0*
T0*>
_class4
20loc:@model/embedding_3/embedding_lookup/25806703*+
_output_shapes
:џџџџџџџџџЅ
-model/embedding_3/embedding_lookup/Identity_1Identity4model/embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџh
model/embedding_2/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
"model/embedding_2/embedding_lookupResourceGather+model_embedding_2_embedding_lookup_25806709model/embedding_2/Cast:y:0*
Tindices0*>
_class4
20loc:@model/embedding_2/embedding_lookup/25806709*+
_output_shapes
:џџџџџџџџџ*
dtype0к
+model/embedding_2/embedding_lookup/IdentityIdentity+model/embedding_2/embedding_lookup:output:0*
T0*>
_class4
20loc:@model/embedding_2/embedding_lookup/25806709*+
_output_shapes
:џџџџџџџџџЅ
-model/embedding_2/embedding_lookup/Identity_1Identity4model/embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџh
model/embedding_1/CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџ
"model/embedding_1/embedding_lookupResourceGather+model_embedding_1_embedding_lookup_25806715model/embedding_1/Cast:y:0*
Tindices0*>
_class4
20loc:@model/embedding_1/embedding_lookup/25806715*+
_output_shapes
:џџџџџџџџџ*
dtype0к
+model/embedding_1/embedding_lookup/IdentityIdentity+model/embedding_1/embedding_lookup:output:0*
T0*>
_class4
20loc:@model/embedding_1/embedding_lookup/25806715*+
_output_shapes
:џџџџџџџџџЅ
-model/embedding_1/embedding_lookup/Identity_1Identity4model/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџf
model/embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџџ
 model/embedding/embedding_lookupResourceGather)model_embedding_embedding_lookup_25806721model/embedding/Cast:y:0*
Tindices0*<
_class2
0.loc:@model/embedding/embedding_lookup/25806721*+
_output_shapes
:џџџџџџџџџ*
dtype0д
)model/embedding/embedding_lookup/IdentityIdentity)model/embedding/embedding_lookup:output:0*
T0*<
_class2
0.loc:@model/embedding/embedding_lookup/25806721*+
_output_shapes
:џџџџџџџџџЁ
+model/embedding/embedding_lookup/Identity_1Identity2model/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџd
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   І
model/flatten/ReshapeReshape4model/embedding/embedding_lookup/Identity_1:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
model/flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ќ
model/flatten_1/ReshapeReshape6model/embedding_1/embedding_lookup/Identity_1:output:0model/flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
model/flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ќ
model/flatten_2/ReshapeReshape6model/embedding_2/embedding_lookup/Identity_1:output:0model/flatten_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
model/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ќ
model/flatten_3/ReshapeReshape6model/embedding_3/embedding_lookup/Identity_1:output:0model/flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
model/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ќ
model/flatten_4/ReshapeReshape6model/embedding_4/embedding_lookup/Identity_1:output:0model/flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
model/flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ќ
model/flatten_5/ReshapeReshape6model/embedding_5/embedding_lookup/Identity_1:output:0model/flatten_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџf
model/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   Ќ
model/flatten_6/ReshapeReshape6model/embedding_6/embedding_lookup/Identity_1:output:0model/flatten_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :ќ
model/concatenate/concatConcatV2model/flatten/Reshape:output:0 model/flatten_1/Reshape:output:0 model/flatten_2/Reshape:output:0 model/flatten_3/Reshape:output:0 model/flatten_4/Reshape:output:0 model/flatten_5/Reshape:output:0 model/flatten_6/Reshape:output:0input_8&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ1
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes
:	1р]*
dtype0
model/dense/MatMulMatMul!model/concatenate/concat:output:0)model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes	
:р]*
dtype0
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]i
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]u
model/dropout/IdentityIdentitymodel/dense/Relu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџр]
&model/quantile_8/MatMul/ReadVariableOpReadVariableOp/model_quantile_8_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_8/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_8/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_8/BiasAddBiasAdd!model/quantile_8/MatMul:product:0/model/quantile_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model/quantile_7/MatMul/ReadVariableOpReadVariableOp/model_quantile_7_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_7/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_7/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_7/BiasAddBiasAdd!model/quantile_7/MatMul:product:0/model/quantile_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model/quantile_6/MatMul/ReadVariableOpReadVariableOp/model_quantile_6_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_6/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_6/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_6/BiasAddBiasAdd!model/quantile_6/MatMul:product:0/model/quantile_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model/quantile_5/MatMul/ReadVariableOpReadVariableOp/model_quantile_5_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_5/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_5/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_5/BiasAddBiasAdd!model/quantile_5/MatMul:product:0/model/quantile_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model/quantile_4/MatMul/ReadVariableOpReadVariableOp/model_quantile_4_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_4/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_4/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_4/BiasAddBiasAdd!model/quantile_4/MatMul:product:0/model/quantile_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model/quantile_3/MatMul/ReadVariableOpReadVariableOp/model_quantile_3_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_3/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_3/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_3/BiasAddBiasAdd!model/quantile_3/MatMul:product:0/model/quantile_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model/quantile_2/MatMul/ReadVariableOpReadVariableOp/model_quantile_2_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_2/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_2/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_2/BiasAddBiasAdd!model/quantile_2/MatMul:product:0/model/quantile_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model/quantile_1/MatMul/ReadVariableOpReadVariableOp/model_quantile_1_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_1/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_1/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_1/BiasAddBiasAdd!model/quantile_1/MatMul:product:0/model/quantile_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
&model/quantile_0/MatMul/ReadVariableOpReadVariableOp/model_quantile_0_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0Є
model/quantile_0/MatMulMatMulmodel/dropout/Identity:output:0.model/quantile_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
'model/quantile_0/BiasAdd/ReadVariableOpReadVariableOp0model_quantile_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Љ
model/quantile_0/BiasAddBiasAdd!model/quantile_0/MatMul:product:0/model/quantile_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџp
IdentityIdentity!model/quantile_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_1Identity!model/quantile_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_2Identity!model/quantile_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_3Identity!model/quantile_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_4Identity!model/quantile_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_5Identity!model/quantile_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_6Identity!model/quantile_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_7Identity!model/quantile_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџr

Identity_8Identity!model/quantile_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџћ
NoOpNoOp#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp!^model/embedding/embedding_lookup#^model/embedding_1/embedding_lookup#^model/embedding_2/embedding_lookup#^model/embedding_3/embedding_lookup#^model/embedding_4/embedding_lookup#^model/embedding_5/embedding_lookup#^model/embedding_6/embedding_lookup(^model/quantile_0/BiasAdd/ReadVariableOp'^model/quantile_0/MatMul/ReadVariableOp(^model/quantile_1/BiasAdd/ReadVariableOp'^model/quantile_1/MatMul/ReadVariableOp(^model/quantile_2/BiasAdd/ReadVariableOp'^model/quantile_2/MatMul/ReadVariableOp(^model/quantile_3/BiasAdd/ReadVariableOp'^model/quantile_3/MatMul/ReadVariableOp(^model/quantile_4/BiasAdd/ReadVariableOp'^model/quantile_4/MatMul/ReadVariableOp(^model/quantile_5/BiasAdd/ReadVariableOp'^model/quantile_5/MatMul/ReadVariableOp(^model/quantile_6/BiasAdd/ReadVariableOp'^model/quantile_6/MatMul/ReadVariableOp(^model/quantile_7/BiasAdd/ReadVariableOp'^model/quantile_7/MatMul/ReadVariableOp(^model/quantile_8/BiasAdd/ReadVariableOp'^model/quantile_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2D
 model/embedding/embedding_lookup model/embedding/embedding_lookup2H
"model/embedding_1/embedding_lookup"model/embedding_1/embedding_lookup2H
"model/embedding_2/embedding_lookup"model/embedding_2/embedding_lookup2H
"model/embedding_3/embedding_lookup"model/embedding_3/embedding_lookup2H
"model/embedding_4/embedding_lookup"model/embedding_4/embedding_lookup2H
"model/embedding_5/embedding_lookup"model/embedding_5/embedding_lookup2H
"model/embedding_6/embedding_lookup"model/embedding_6/embedding_lookup2R
'model/quantile_0/BiasAdd/ReadVariableOp'model/quantile_0/BiasAdd/ReadVariableOp2P
&model/quantile_0/MatMul/ReadVariableOp&model/quantile_0/MatMul/ReadVariableOp2R
'model/quantile_1/BiasAdd/ReadVariableOp'model/quantile_1/BiasAdd/ReadVariableOp2P
&model/quantile_1/MatMul/ReadVariableOp&model/quantile_1/MatMul/ReadVariableOp2R
'model/quantile_2/BiasAdd/ReadVariableOp'model/quantile_2/BiasAdd/ReadVariableOp2P
&model/quantile_2/MatMul/ReadVariableOp&model/quantile_2/MatMul/ReadVariableOp2R
'model/quantile_3/BiasAdd/ReadVariableOp'model/quantile_3/BiasAdd/ReadVariableOp2P
&model/quantile_3/MatMul/ReadVariableOp&model/quantile_3/MatMul/ReadVariableOp2R
'model/quantile_4/BiasAdd/ReadVariableOp'model/quantile_4/BiasAdd/ReadVariableOp2P
&model/quantile_4/MatMul/ReadVariableOp&model/quantile_4/MatMul/ReadVariableOp2R
'model/quantile_5/BiasAdd/ReadVariableOp'model/quantile_5/BiasAdd/ReadVariableOp2P
&model/quantile_5/MatMul/ReadVariableOp&model/quantile_5/MatMul/ReadVariableOp2R
'model/quantile_6/BiasAdd/ReadVariableOp'model/quantile_6/BiasAdd/ReadVariableOp2P
&model/quantile_6/MatMul/ReadVariableOp&model/quantile_6/MatMul/ReadVariableOp2R
'model/quantile_7/BiasAdd/ReadVariableOp'model/quantile_7/BiasAdd/ReadVariableOp2P
&model/quantile_7/MatMul/ReadVariableOp&model/quantile_7/MatMul/ReadVariableOp2R
'model/quantile_8/BiasAdd/ReadVariableOp'model/quantile_8/BiasAdd/ReadVariableOp2P
&model/quantile_8/MatMul/ReadVariableOp&model/quantile_8/MatMul/ReadVariableOp:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
П
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_25806944

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_25806960

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_1_layer_call_and_return_conditional_losses_25806904

inputs+
embedding_lookup_25806898:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25806898Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25806898*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25806898*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ш"
Х
(__inference_model_layer_call_fn_25808198
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
unknown:
	unknown_0:8
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:	1р]
	unknown_7:	р]
	unknown_8:	р]
	unknown_9:

unknown_10:	р]

unknown_11:

unknown_12:	р]

unknown_13:

unknown_14:	р]

unknown_15:

unknown_16:	р]

unknown_17:

unknown_18:	р]

unknown_19:

unknown_20:	р]

unknown_21:

unknown_22:	р]

unknown_23:

unknown_24:	р]

unknown_25:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*.
Tin'
%2#*
Tout
2	*
_collective_manager_ids
 *С
_output_shapesЎ
Ћ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*=
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_25807390o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Я	
њ
H__inference_quantile_4_layer_call_and_return_conditional_losses_25808928

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Э

-__inference_quantile_5_layer_call_fn_25808937

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_5_layer_call_and_return_conditional_losses_25807082o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_7_layer_call_and_return_conditional_losses_25807050

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Э

-__inference_quantile_4_layer_call_fn_25808918

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_4_layer_call_and_return_conditional_losses_25807098o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Э

-__inference_quantile_0_layer_call_fn_25808842

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_0_layer_call_and_return_conditional_losses_25807162o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_0_layer_call_and_return_conditional_losses_25807162

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Ћ
H
,__inference_flatten_2_layer_call_fn_25808711

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_25806944`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_25806936

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ

d
E__inference_dropout_layer_call_and_return_conditional_losses_25808828

inputs
identityR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]Q
dropout/ShapeShapeinputs*
T0*
_output_shapes
::эЯ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]*
dtype0*

seed[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?Ї
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]T
dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    
dropout/SelectV2SelectV2dropout/GreaterEqual:z:0dropout/Mul:z:0dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]b
IdentityIdentitydropout/SelectV2:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:џџџџџџџџџр]:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
П
c
G__inference_flatten_2_layer_call_and_return_conditional_losses_25808717

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
оЉ
Ї?
$__inference__traced_restore_25809979
file_prefix7
%assignvariableop_embedding_embeddings:;
)assignvariableop_1_embedding_1_embeddings:;
)assignvariableop_2_embedding_2_embeddings:;
)assignvariableop_3_embedding_3_embeddings:;
)assignvariableop_4_embedding_4_embeddings:;
)assignvariableop_5_embedding_5_embeddings:8;
)assignvariableop_6_embedding_6_embeddings:2
assignvariableop_7_dense_kernel:	1р],
assignvariableop_8_dense_bias:	р]7
$assignvariableop_9_quantile_0_kernel:	р]1
#assignvariableop_10_quantile_0_bias:8
%assignvariableop_11_quantile_1_kernel:	р]1
#assignvariableop_12_quantile_1_bias:8
%assignvariableop_13_quantile_2_kernel:	р]1
#assignvariableop_14_quantile_2_bias:8
%assignvariableop_15_quantile_3_kernel:	р]1
#assignvariableop_16_quantile_3_bias:8
%assignvariableop_17_quantile_4_kernel:	р]1
#assignvariableop_18_quantile_4_bias:8
%assignvariableop_19_quantile_5_kernel:	р]1
#assignvariableop_20_quantile_5_bias:8
%assignvariableop_21_quantile_6_kernel:	р]1
#assignvariableop_22_quantile_6_bias:8
%assignvariableop_23_quantile_7_kernel:	р]1
#assignvariableop_24_quantile_7_bias:8
%assignvariableop_25_quantile_8_kernel:	р]1
#assignvariableop_26_quantile_8_bias:'
assignvariableop_27_iteration:	 +
!assignvariableop_28_learning_rate: A
/assignvariableop_29_adam_m_embedding_embeddings:A
/assignvariableop_30_adam_v_embedding_embeddings:C
1assignvariableop_31_adam_m_embedding_1_embeddings:C
1assignvariableop_32_adam_v_embedding_1_embeddings:C
1assignvariableop_33_adam_m_embedding_2_embeddings:C
1assignvariableop_34_adam_v_embedding_2_embeddings:C
1assignvariableop_35_adam_m_embedding_3_embeddings:C
1assignvariableop_36_adam_v_embedding_3_embeddings:C
1assignvariableop_37_adam_m_embedding_4_embeddings:C
1assignvariableop_38_adam_v_embedding_4_embeddings:C
1assignvariableop_39_adam_m_embedding_5_embeddings:8C
1assignvariableop_40_adam_v_embedding_5_embeddings:8C
1assignvariableop_41_adam_m_embedding_6_embeddings:C
1assignvariableop_42_adam_v_embedding_6_embeddings::
'assignvariableop_43_adam_m_dense_kernel:	1р]:
'assignvariableop_44_adam_v_dense_kernel:	1р]4
%assignvariableop_45_adam_m_dense_bias:	р]4
%assignvariableop_46_adam_v_dense_bias:	р]?
,assignvariableop_47_adam_m_quantile_0_kernel:	р]?
,assignvariableop_48_adam_v_quantile_0_kernel:	р]8
*assignvariableop_49_adam_m_quantile_0_bias:8
*assignvariableop_50_adam_v_quantile_0_bias:?
,assignvariableop_51_adam_m_quantile_1_kernel:	р]?
,assignvariableop_52_adam_v_quantile_1_kernel:	р]8
*assignvariableop_53_adam_m_quantile_1_bias:8
*assignvariableop_54_adam_v_quantile_1_bias:?
,assignvariableop_55_adam_m_quantile_2_kernel:	р]?
,assignvariableop_56_adam_v_quantile_2_kernel:	р]8
*assignvariableop_57_adam_m_quantile_2_bias:8
*assignvariableop_58_adam_v_quantile_2_bias:?
,assignvariableop_59_adam_m_quantile_3_kernel:	р]?
,assignvariableop_60_adam_v_quantile_3_kernel:	р]8
*assignvariableop_61_adam_m_quantile_3_bias:8
*assignvariableop_62_adam_v_quantile_3_bias:?
,assignvariableop_63_adam_m_quantile_4_kernel:	р]?
,assignvariableop_64_adam_v_quantile_4_kernel:	р]8
*assignvariableop_65_adam_m_quantile_4_bias:8
*assignvariableop_66_adam_v_quantile_4_bias:?
,assignvariableop_67_adam_m_quantile_5_kernel:	р]?
,assignvariableop_68_adam_v_quantile_5_kernel:	р]8
*assignvariableop_69_adam_m_quantile_5_bias:8
*assignvariableop_70_adam_v_quantile_5_bias:?
,assignvariableop_71_adam_m_quantile_6_kernel:	р]?
,assignvariableop_72_adam_v_quantile_6_kernel:	р]8
*assignvariableop_73_adam_m_quantile_6_bias:8
*assignvariableop_74_adam_v_quantile_6_bias:?
,assignvariableop_75_adam_m_quantile_7_kernel:	р]?
,assignvariableop_76_adam_v_quantile_7_kernel:	р]8
*assignvariableop_77_adam_m_quantile_7_bias:8
*assignvariableop_78_adam_v_quantile_7_bias:?
,assignvariableop_79_adam_m_quantile_8_kernel:	р]?
,assignvariableop_80_adam_v_quantile_8_kernel:	р]8
*assignvariableop_81_adam_m_quantile_8_bias:8
*assignvariableop_82_adam_v_quantile_8_bias:%
assignvariableop_83_total_9: %
assignvariableop_84_count_9: %
assignvariableop_85_total_8: %
assignvariableop_86_count_8: %
assignvariableop_87_total_7: %
assignvariableop_88_count_7: %
assignvariableop_89_total_6: %
assignvariableop_90_count_6: %
assignvariableop_91_total_5: %
assignvariableop_92_count_5: %
assignvariableop_93_total_4: %
assignvariableop_94_count_4: %
assignvariableop_95_total_3: %
assignvariableop_96_count_3: %
assignvariableop_97_total_2: %
assignvariableop_98_count_2: %
assignvariableop_99_total_1: &
assignvariableop_100_count_1: $
assignvariableop_101_total: $
assignvariableop_102_count: 
identity_104ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_100ЂAssignVariableOp_101ЂAssignVariableOp_102ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_40ЂAssignVariableOp_41ЂAssignVariableOp_42ЂAssignVariableOp_43ЂAssignVariableOp_44ЂAssignVariableOp_45ЂAssignVariableOp_46ЂAssignVariableOp_47ЂAssignVariableOp_48ЂAssignVariableOp_49ЂAssignVariableOp_5ЂAssignVariableOp_50ЂAssignVariableOp_51ЂAssignVariableOp_52ЂAssignVariableOp_53ЂAssignVariableOp_54ЂAssignVariableOp_55ЂAssignVariableOp_56ЂAssignVariableOp_57ЂAssignVariableOp_58ЂAssignVariableOp_59ЂAssignVariableOp_6ЂAssignVariableOp_60ЂAssignVariableOp_61ЂAssignVariableOp_62ЂAssignVariableOp_63ЂAssignVariableOp_64ЂAssignVariableOp_65ЂAssignVariableOp_66ЂAssignVariableOp_67ЂAssignVariableOp_68ЂAssignVariableOp_69ЂAssignVariableOp_7ЂAssignVariableOp_70ЂAssignVariableOp_71ЂAssignVariableOp_72ЂAssignVariableOp_73ЂAssignVariableOp_74ЂAssignVariableOp_75ЂAssignVariableOp_76ЂAssignVariableOp_77ЂAssignVariableOp_78ЂAssignVariableOp_79ЂAssignVariableOp_8ЂAssignVariableOp_80ЂAssignVariableOp_81ЂAssignVariableOp_82ЂAssignVariableOp_83ЂAssignVariableOp_84ЂAssignVariableOp_85ЂAssignVariableOp_86ЂAssignVariableOp_87ЂAssignVariableOp_88ЂAssignVariableOp_89ЂAssignVariableOp_9ЂAssignVariableOp_90ЂAssignVariableOp_91ЂAssignVariableOp_92ЂAssignVariableOp_93ЂAssignVariableOp_94ЂAssignVariableOp_95ЂAssignVariableOp_96ЂAssignVariableOp_97ЂAssignVariableOp_98ЂAssignVariableOp_99,
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*П+
valueЕ+BВ+hB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHУ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*х
valueлBиhB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Љ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ж
_output_shapesЃ
 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*v
dtypesl
j2h	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOpAssignVariableOp%assignvariableop_embedding_embeddingsIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_1AssignVariableOp)assignvariableop_1_embedding_1_embeddingsIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_2AssignVariableOp)assignvariableop_2_embedding_2_embeddingsIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_3AssignVariableOp)assignvariableop_3_embedding_3_embeddingsIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_4AssignVariableOp)assignvariableop_4_embedding_4_embeddingsIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_5AssignVariableOp)assignvariableop_5_embedding_5_embeddingsIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_6AssignVariableOp)assignvariableop_6_embedding_6_embeddingsIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_8AssignVariableOpassignvariableop_8_dense_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Л
AssignVariableOp_9AssignVariableOp$assignvariableop_9_quantile_0_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_10AssignVariableOp#assignvariableop_10_quantile_0_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_11AssignVariableOp%assignvariableop_11_quantile_1_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOp#assignvariableop_12_quantile_1_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_13AssignVariableOp%assignvariableop_13_quantile_2_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOp#assignvariableop_14_quantile_2_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_15AssignVariableOp%assignvariableop_15_quantile_3_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_16AssignVariableOp#assignvariableop_16_quantile_3_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_17AssignVariableOp%assignvariableop_17_quantile_4_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_18AssignVariableOp#assignvariableop_18_quantile_4_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_19AssignVariableOp%assignvariableop_19_quantile_5_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_20AssignVariableOp#assignvariableop_20_quantile_5_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_21AssignVariableOp%assignvariableop_21_quantile_6_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_22AssignVariableOp#assignvariableop_22_quantile_6_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_23AssignVariableOp%assignvariableop_23_quantile_7_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_24AssignVariableOp#assignvariableop_24_quantile_7_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_25AssignVariableOp%assignvariableop_25_quantile_8_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_26AssignVariableOp#assignvariableop_26_quantile_8_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0	*
_output_shapes
:Ж
AssignVariableOp_27AssignVariableOpassignvariableop_27_iterationIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_28AssignVariableOp!assignvariableop_28_learning_rateIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_29AssignVariableOp/assignvariableop_29_adam_m_embedding_embeddingsIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:Ш
AssignVariableOp_30AssignVariableOp/assignvariableop_30_adam_v_embedding_embeddingsIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_31AssignVariableOp1assignvariableop_31_adam_m_embedding_1_embeddingsIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_32AssignVariableOp1assignvariableop_32_adam_v_embedding_1_embeddingsIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_33AssignVariableOp1assignvariableop_33_adam_m_embedding_2_embeddingsIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_34AssignVariableOp1assignvariableop_34_adam_v_embedding_2_embeddingsIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_35AssignVariableOp1assignvariableop_35_adam_m_embedding_3_embeddingsIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_36AssignVariableOp1assignvariableop_36_adam_v_embedding_3_embeddingsIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_37AssignVariableOp1assignvariableop_37_adam_m_embedding_4_embeddingsIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_38AssignVariableOp1assignvariableop_38_adam_v_embedding_4_embeddingsIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_39AssignVariableOp1assignvariableop_39_adam_m_embedding_5_embeddingsIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_40AssignVariableOp1assignvariableop_40_adam_v_embedding_5_embeddingsIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_41AssignVariableOp1assignvariableop_41_adam_m_embedding_6_embeddingsIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_42AssignVariableOp1assignvariableop_42_adam_v_embedding_6_embeddingsIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_m_dense_kernelIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_v_dense_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_45AssignVariableOp%assignvariableop_45_adam_m_dense_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:О
AssignVariableOp_46AssignVariableOp%assignvariableop_46_adam_v_dense_biasIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_47AssignVariableOp,assignvariableop_47_adam_m_quantile_0_kernelIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_48AssignVariableOp,assignvariableop_48_adam_v_quantile_0_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_49AssignVariableOp*assignvariableop_49_adam_m_quantile_0_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_50AssignVariableOp*assignvariableop_50_adam_v_quantile_0_biasIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_51AssignVariableOp,assignvariableop_51_adam_m_quantile_1_kernelIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_52AssignVariableOp,assignvariableop_52_adam_v_quantile_1_kernelIdentity_52:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_m_quantile_1_biasIdentity_53:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_54AssignVariableOp*assignvariableop_54_adam_v_quantile_1_biasIdentity_54:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_55AssignVariableOp,assignvariableop_55_adam_m_quantile_2_kernelIdentity_55:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_v_quantile_2_kernelIdentity_56:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_57AssignVariableOp*assignvariableop_57_adam_m_quantile_2_biasIdentity_57:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_58AssignVariableOp*assignvariableop_58_adam_v_quantile_2_biasIdentity_58:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_59AssignVariableOp,assignvariableop_59_adam_m_quantile_3_kernelIdentity_59:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_60AssignVariableOp,assignvariableop_60_adam_v_quantile_3_kernelIdentity_60:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_61AssignVariableOp*assignvariableop_61_adam_m_quantile_3_biasIdentity_61:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_62AssignVariableOp*assignvariableop_62_adam_v_quantile_3_biasIdentity_62:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_63AssignVariableOp,assignvariableop_63_adam_m_quantile_4_kernelIdentity_63:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_64AssignVariableOp,assignvariableop_64_adam_v_quantile_4_kernelIdentity_64:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_m_quantile_4_biasIdentity_65:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_66AssignVariableOp*assignvariableop_66_adam_v_quantile_4_biasIdentity_66:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_67AssignVariableOp,assignvariableop_67_adam_m_quantile_5_kernelIdentity_67:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_68AssignVariableOp,assignvariableop_68_adam_v_quantile_5_kernelIdentity_68:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_m_quantile_5_biasIdentity_69:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_70AssignVariableOp*assignvariableop_70_adam_v_quantile_5_biasIdentity_70:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_71AssignVariableOp,assignvariableop_71_adam_m_quantile_6_kernelIdentity_71:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_72AssignVariableOp,assignvariableop_72_adam_v_quantile_6_kernelIdentity_72:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_m_quantile_6_biasIdentity_73:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_74AssignVariableOp*assignvariableop_74_adam_v_quantile_6_biasIdentity_74:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_75AssignVariableOp,assignvariableop_75_adam_m_quantile_7_kernelIdentity_75:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_76AssignVariableOp,assignvariableop_76_adam_v_quantile_7_kernelIdentity_76:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_77AssignVariableOp*assignvariableop_77_adam_m_quantile_7_biasIdentity_77:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_78AssignVariableOp*assignvariableop_78_adam_v_quantile_7_biasIdentity_78:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_79AssignVariableOp,assignvariableop_79_adam_m_quantile_8_kernelIdentity_79:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:Х
AssignVariableOp_80AssignVariableOp,assignvariableop_80_adam_v_quantile_8_kernelIdentity_80:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_81AssignVariableOp*assignvariableop_81_adam_m_quantile_8_biasIdentity_81:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_82AssignVariableOp*assignvariableop_82_adam_v_quantile_8_biasIdentity_82:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_83AssignVariableOpassignvariableop_83_total_9Identity_83:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_84AssignVariableOpassignvariableop_84_count_9Identity_84:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_85AssignVariableOpassignvariableop_85_total_8Identity_85:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_86AssignVariableOpassignvariableop_86_count_8Identity_86:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_87AssignVariableOpassignvariableop_87_total_7Identity_87:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_88AssignVariableOpassignvariableop_88_count_7Identity_88:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_89AssignVariableOpassignvariableop_89_total_6Identity_89:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_90AssignVariableOpassignvariableop_90_count_6Identity_90:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_91AssignVariableOpassignvariableop_91_total_5Identity_91:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_92AssignVariableOpassignvariableop_92_count_5Identity_92:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_93AssignVariableOpassignvariableop_93_total_4Identity_93:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_94AssignVariableOpassignvariableop_94_count_4Identity_94:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_95AssignVariableOpassignvariableop_95_total_3Identity_95:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_96AssignVariableOpassignvariableop_96_count_3Identity_96:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_97AssignVariableOpassignvariableop_97_total_2Identity_97:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_98AssignVariableOpassignvariableop_98_count_2Identity_98:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_99AssignVariableOpassignvariableop_99_total_1Identity_99:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_100AssignVariableOpassignvariableop_100_count_1Identity_100:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_101AssignVariableOpassignvariableop_101_totalIdentity_101:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0a
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_102AssignVariableOpassignvariableop_102_countIdentity_102:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ­
Identity_103Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: Y
Identity_104IdentityIdentity_103:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*"
_acd_function_control_output(*
_output_shapes
 "%
identity_104Identity_104:output:0*х
_input_shapesг
а: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_992(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Н
a
E__inference_flatten_layer_call_and_return_conditional_losses_25808695

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ	
І
G__inference_embedding_layer_call_and_return_conditional_losses_25808582

inputs+
embedding_lookup_25808576:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25808576Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25808576*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25808576*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_25806952

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_3_layer_call_and_return_conditional_losses_25808633

inputs+
embedding_lookup_25808627:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25808627Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25808627*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25808627*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ

і
C__inference_dense_layer_call_and_return_conditional_losses_25807004

inputs1
matmul_readvariableop_resource:	1р].
biasadd_readvariableop_resource:	р]
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	1р]*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:р]*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџр]w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ1
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_0_layer_call_and_return_conditional_losses_25808852

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
ш"
Х
(__inference_model_layer_call_fn_25808280
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
unknown:
	unknown_0:8
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:	1р]
	unknown_7:	р]
	unknown_8:	р]
	unknown_9:

unknown_10:	р]

unknown_11:

unknown_12:	р]

unknown_13:

unknown_14:	р]

unknown_15:

unknown_16:	р]

unknown_17:

unknown_18:	р]

unknown_19:

unknown_20:	р]

unknown_21:

unknown_22:	р]

unknown_23:

unknown_24:	р]

unknown_25:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*.
Tin'
%2#*
Tout
2	*
_collective_manager_ids
 *С
_output_shapesЎ
Ћ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*=
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_25807571o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Я	
њ
H__inference_quantile_6_layer_call_and_return_conditional_losses_25808966

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Ї
Ч
I__inference_concatenate_layer_call_and_return_conditional_losses_25806991

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Б
concatConcatV2inputsinputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ1W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э

-__inference_quantile_7_layer_call_fn_25808975

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_7_layer_call_and_return_conditional_losses_25807050o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Э

-__inference_quantile_2_layer_call_fn_25808880

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_2_layer_call_and_return_conditional_losses_25807130o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_5_layer_call_and_return_conditional_losses_25807082

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Ћ
H
,__inference_flatten_3_layer_call_fn_25808722

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_25806952`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ћ
H
,__inference_flatten_5_layer_call_fn_25808744

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_25806968`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_4_layer_call_and_return_conditional_losses_25808650

inputs+
embedding_lookup_25808644:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25808644Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25808644*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25808644*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_4_layer_call_and_return_conditional_losses_25807098

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
П
c
G__inference_flatten_1_layer_call_and_return_conditional_losses_25808706

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_5_layer_call_and_return_conditional_losses_25806848

inputs+
embedding_lookup_25806842:8
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25806842Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25806842*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25806842*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_2_layer_call_and_return_conditional_losses_25806890

inputs+
embedding_lookup_25806884:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25806884Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25806884*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25806884*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_1_layer_call_and_return_conditional_losses_25808871

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_6_layer_call_and_return_conditional_losses_25807066

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Э

-__inference_quantile_3_layer_call_fn_25808899

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_3_layer_call_and_return_conditional_losses_25807114o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_3_layer_call_and_return_conditional_losses_25806876

inputs+
embedding_lookup_25806870:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25806870Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25806870*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25806870*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ф

(__inference_dense_layer_call_fn_25808795

inputs
unknown:	1р]
	unknown_0:	р]
identityЂStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_25807004p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџр]`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ1
 
_user_specified_nameinputs
ѕ
ж
C__inference_model_layer_call_and_return_conditional_losses_25807390

inputs
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7&
embedding_6_25807301:&
embedding_5_25807304:8&
embedding_4_25807307:&
embedding_3_25807310:&
embedding_2_25807313:&
embedding_1_25807316:$
embedding_25807319:!
dense_25807330:	1р]
dense_25807332:	р]&
quantile_8_25807336:	р]!
quantile_8_25807338:&
quantile_7_25807341:	р]!
quantile_7_25807343:&
quantile_6_25807346:	р]!
quantile_6_25807348:&
quantile_5_25807351:	р]!
quantile_5_25807353:&
quantile_4_25807356:	р]!
quantile_4_25807358:&
quantile_3_25807361:	р]!
quantile_3_25807363:&
quantile_2_25807366:	р]!
quantile_2_25807368:&
quantile_1_25807371:	р]!
quantile_1_25807373:&
quantile_0_25807376:	р]!
quantile_0_25807378:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8Ђdense/StatefulPartitionedCallЂdropout/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂ#embedding_1/StatefulPartitionedCallЂ#embedding_2/StatefulPartitionedCallЂ#embedding_3/StatefulPartitionedCallЂ#embedding_4/StatefulPartitionedCallЂ#embedding_5/StatefulPartitionedCallЂ#embedding_6/StatefulPartitionedCallЂ"quantile_0/StatefulPartitionedCallЂ"quantile_1/StatefulPartitionedCallЂ"quantile_2/StatefulPartitionedCallЂ"quantile_3/StatefulPartitionedCallЂ"quantile_4/StatefulPartitionedCallЂ"quantile_5/StatefulPartitionedCallЂ"quantile_6/StatefulPartitionedCallЂ"quantile_7/StatefulPartitionedCallЂ"quantile_8/StatefulPartitionedCall№
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinputs_6embedding_6_25807301*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_6_layer_call_and_return_conditional_losses_25806834№
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallinputs_5embedding_5_25807304*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_5_layer_call_and_return_conditional_losses_25806848№
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallinputs_4embedding_4_25807307*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_4_layer_call_and_return_conditional_losses_25806862№
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallinputs_3embedding_3_25807310*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_3_layer_call_and_return_conditional_losses_25806876№
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinputs_2embedding_2_25807313*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_25806890№
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_1_25807316*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_25806904ш
!embedding/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_25807319*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_25806918м
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_25806928т
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_25806936т
flatten_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_25806944т
flatten_3/PartitionedCallPartitionedCall,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_25806952т
flatten_4/PartitionedCallPartitionedCall,embedding_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_25806960т
flatten_5/PartitionedCallPartitionedCall,embedding_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_25806968т
flatten_6/PartitionedCallPartitionedCall,embedding_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_25806976У
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0inputs_7*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_25806991
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_25807330dense_25807332*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_25807004щ
dropout/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_25807022 
"quantile_8/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_8_25807336quantile_8_25807338*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_8_layer_call_and_return_conditional_losses_25807034 
"quantile_7/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_7_25807341quantile_7_25807343*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_7_layer_call_and_return_conditional_losses_25807050 
"quantile_6/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_6_25807346quantile_6_25807348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_6_layer_call_and_return_conditional_losses_25807066 
"quantile_5/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_5_25807351quantile_5_25807353*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_5_layer_call_and_return_conditional_losses_25807082 
"quantile_4/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_4_25807356quantile_4_25807358*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_4_layer_call_and_return_conditional_losses_25807098 
"quantile_3/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_3_25807361quantile_3_25807363*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_3_layer_call_and_return_conditional_losses_25807114 
"quantile_2/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_2_25807366quantile_2_25807368*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_2_layer_call_and_return_conditional_losses_25807130 
"quantile_1/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_1_25807371quantile_1_25807373*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_1_layer_call_and_return_conditional_losses_25807146 
"quantile_0/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0quantile_0_25807376quantile_0_25807378*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_0_layer_call_and_return_conditional_losses_25807162z
IdentityIdentity+quantile_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_1Identity+quantile_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_2Identity+quantile_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_3Identity+quantile_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_4Identity+quantile_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_5Identity+quantile_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_6Identity+quantile_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_7Identity+quantile_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_8Identity+quantile_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџн
NoOpNoOp^dense/StatefulPartitionedCall ^dropout/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall#^quantile_0/StatefulPartitionedCall#^quantile_1/StatefulPartitionedCall#^quantile_2/StatefulPartitionedCall#^quantile_3/StatefulPartitionedCall#^quantile_4/StatefulPartitionedCall#^quantile_5/StatefulPartitionedCall#^quantile_6/StatefulPartitionedCall#^quantile_7/StatefulPartitionedCall#^quantile_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2H
"quantile_0/StatefulPartitionedCall"quantile_0/StatefulPartitionedCall2H
"quantile_1/StatefulPartitionedCall"quantile_1/StatefulPartitionedCall2H
"quantile_2/StatefulPartitionedCall"quantile_2/StatefulPartitionedCall2H
"quantile_3/StatefulPartitionedCall"quantile_3/StatefulPartitionedCall2H
"quantile_4/StatefulPartitionedCall"quantile_4/StatefulPartitionedCall2H
"quantile_5/StatefulPartitionedCall"quantile_5/StatefulPartitionedCall2H
"quantile_6/StatefulPartitionedCall"quantile_6/StatefulPartitionedCall2H
"quantile_7/StatefulPartitionedCall"quantile_7/StatefulPartitionedCall2H
"quantile_8/StatefulPartitionedCall"quantile_8/StatefulPartitionedCall:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ

C__inference_model_layer_call_and_return_conditional_losses_25808426
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_77
%embedding_6_embedding_lookup_25808291:7
%embedding_5_embedding_lookup_25808297:87
%embedding_4_embedding_lookup_25808303:7
%embedding_3_embedding_lookup_25808309:7
%embedding_2_embedding_lookup_25808315:7
%embedding_1_embedding_lookup_25808321:5
#embedding_embedding_lookup_25808327:7
$dense_matmul_readvariableop_resource:	1р]4
%dense_biasadd_readvariableop_resource:	р]<
)quantile_8_matmul_readvariableop_resource:	р]8
*quantile_8_biasadd_readvariableop_resource:<
)quantile_7_matmul_readvariableop_resource:	р]8
*quantile_7_biasadd_readvariableop_resource:<
)quantile_6_matmul_readvariableop_resource:	р]8
*quantile_6_biasadd_readvariableop_resource:<
)quantile_5_matmul_readvariableop_resource:	р]8
*quantile_5_biasadd_readvariableop_resource:<
)quantile_4_matmul_readvariableop_resource:	р]8
*quantile_4_biasadd_readvariableop_resource:<
)quantile_3_matmul_readvariableop_resource:	р]8
*quantile_3_biasadd_readvariableop_resource:<
)quantile_2_matmul_readvariableop_resource:	р]8
*quantile_2_biasadd_readvariableop_resource:<
)quantile_1_matmul_readvariableop_resource:	р]8
*quantile_1_biasadd_readvariableop_resource:<
)quantile_0_matmul_readvariableop_resource:	р]8
*quantile_0_biasadd_readvariableop_resource:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8Ђdense/BiasAdd/ReadVariableOpЂdense/MatMul/ReadVariableOpЂembedding/embedding_lookupЂembedding_1/embedding_lookupЂembedding_2/embedding_lookupЂembedding_3/embedding_lookupЂembedding_4/embedding_lookupЂembedding_5/embedding_lookupЂembedding_6/embedding_lookupЂ!quantile_0/BiasAdd/ReadVariableOpЂ quantile_0/MatMul/ReadVariableOpЂ!quantile_1/BiasAdd/ReadVariableOpЂ quantile_1/MatMul/ReadVariableOpЂ!quantile_2/BiasAdd/ReadVariableOpЂ quantile_2/MatMul/ReadVariableOpЂ!quantile_3/BiasAdd/ReadVariableOpЂ quantile_3/MatMul/ReadVariableOpЂ!quantile_4/BiasAdd/ReadVariableOpЂ quantile_4/MatMul/ReadVariableOpЂ!quantile_5/BiasAdd/ReadVariableOpЂ quantile_5/MatMul/ReadVariableOpЂ!quantile_6/BiasAdd/ReadVariableOpЂ quantile_6/MatMul/ReadVariableOpЂ!quantile_7/BiasAdd/ReadVariableOpЂ quantile_7/MatMul/ReadVariableOpЂ!quantile_8/BiasAdd/ReadVariableOpЂ quantile_8/MatMul/ReadVariableOpc
embedding_6/CastCastinputs_6*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_6/embedding_lookupResourceGather%embedding_6_embedding_lookup_25808291embedding_6/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_6/embedding_lookup/25808291*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_6/embedding_lookup/IdentityIdentity%embedding_6/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_6/embedding_lookup/25808291*+
_output_shapes
:џџџџџџџџџ
'embedding_6/embedding_lookup/Identity_1Identity.embedding_6/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_5/CastCastinputs_5*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_5/embedding_lookupResourceGather%embedding_5_embedding_lookup_25808297embedding_5/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_5/embedding_lookup/25808297*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_5/embedding_lookup/IdentityIdentity%embedding_5/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_5/embedding_lookup/25808297*+
_output_shapes
:џџџџџџџџџ
'embedding_5/embedding_lookup/Identity_1Identity.embedding_5/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_4/CastCastinputs_4*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_4/embedding_lookupResourceGather%embedding_4_embedding_lookup_25808303embedding_4/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_4/embedding_lookup/25808303*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_4/embedding_lookup/IdentityIdentity%embedding_4/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_4/embedding_lookup/25808303*+
_output_shapes
:џџџџџџџџџ
'embedding_4/embedding_lookup/Identity_1Identity.embedding_4/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_3/CastCastinputs_3*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_3/embedding_lookupResourceGather%embedding_3_embedding_lookup_25808309embedding_3/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_3/embedding_lookup/25808309*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_3/embedding_lookup/IdentityIdentity%embedding_3/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_3/embedding_lookup/25808309*+
_output_shapes
:џџџџџџџџџ
'embedding_3/embedding_lookup/Identity_1Identity.embedding_3/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_2/CastCastinputs_2*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_2/embedding_lookupResourceGather%embedding_2_embedding_lookup_25808315embedding_2/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_2/embedding_lookup/25808315*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_2/embedding_lookup/IdentityIdentity%embedding_2/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_2/embedding_lookup/25808315*+
_output_shapes
:џџџџџџџџџ
'embedding_2/embedding_lookup/Identity_1Identity.embedding_2/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџc
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџя
embedding_1/embedding_lookupResourceGather%embedding_1_embedding_lookup_25808321embedding_1/Cast:y:0*
Tindices0*8
_class.
,*loc:@embedding_1/embedding_lookup/25808321*+
_output_shapes
:џџџџџџџџџ*
dtype0Ш
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0*
T0*8
_class.
,*loc:@embedding_1/embedding_lookup/25808321*+
_output_shapes
:џџџџџџџџџ
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџa
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџч
embedding/embedding_lookupResourceGather#embedding_embedding_lookup_25808327embedding/Cast:y:0*
Tindices0*6
_class,
*(loc:@embedding/embedding_lookup/25808327*+
_output_shapes
:џџџџџџџџџ*
dtype0Т
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0*
T0*6
_class,
*(loc:@embedding/embedding_lookup/25808327*+
_output_shapes
:џџџџџџџџџ
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџ^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten/ReshapeReshape.embedding/embedding_lookup/Identity_1:output:0flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_1/ReshapeReshape0embedding_1/embedding_lookup/Identity_1:output:0flatten_1/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_2/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_2/ReshapeReshape0embedding_2/embedding_lookup/Identity_1:output:0flatten_2/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_3/ReshapeReshape0embedding_3/embedding_lookup/Identity_1:output:0flatten_3/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_4/ReshapeReshape0embedding_4/embedding_lookup/Identity_1:output:0flatten_4/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_5/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_5/ReshapeReshape0embedding_5/embedding_lookup/Identity_1:output:0flatten_5/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   
flatten_6/ReshapeReshape0embedding_6/embedding_lookup/Identity_1:output:0flatten_6/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџY
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Ч
concatenate/concatConcatV2flatten/Reshape:output:0flatten_1/Reshape:output:0flatten_2/Reshape:output:0flatten_3/Reshape:output:0flatten_4/Reshape:output:0flatten_5/Reshape:output:0flatten_6/Reshape:output:0inputs_7 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ1
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	1р]*
dtype0
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:р]*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]Z
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   @
dropout/dropout/MulMuldense/Relu:activations:0dropout/dropout/Const:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]k
dropout/dropout/ShapeShapedense/Relu:activations:0*
T0*
_output_shapes
::эЯЉ
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]*
dtype0*

seedc
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *   ?П
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]\
dropout/dropout/Const_1Const*
_output_shapes
: *
dtype0*
valueB
 *    Д
dropout/dropout/SelectV2SelectV2 dropout/dropout/GreaterEqual:z:0dropout/dropout/Mul:z:0 dropout/dropout/Const_1:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]
 quantile_8/MatMul/ReadVariableOpReadVariableOp)quantile_8_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_8/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_8/BiasAdd/ReadVariableOpReadVariableOp*quantile_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_8/BiasAddBiasAddquantile_8/MatMul:product:0)quantile_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_7/MatMul/ReadVariableOpReadVariableOp)quantile_7_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_7/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_7/BiasAdd/ReadVariableOpReadVariableOp*quantile_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_7/BiasAddBiasAddquantile_7/MatMul:product:0)quantile_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_6/MatMul/ReadVariableOpReadVariableOp)quantile_6_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_6/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_6/BiasAdd/ReadVariableOpReadVariableOp*quantile_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_6/BiasAddBiasAddquantile_6/MatMul:product:0)quantile_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_5/MatMul/ReadVariableOpReadVariableOp)quantile_5_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_5/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_5/BiasAdd/ReadVariableOpReadVariableOp*quantile_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_5/BiasAddBiasAddquantile_5/MatMul:product:0)quantile_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_4/MatMul/ReadVariableOpReadVariableOp)quantile_4_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_4/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_4/BiasAdd/ReadVariableOpReadVariableOp*quantile_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_4/BiasAddBiasAddquantile_4/MatMul:product:0)quantile_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_3/MatMul/ReadVariableOpReadVariableOp)quantile_3_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_3/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_3/BiasAdd/ReadVariableOpReadVariableOp*quantile_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_3/BiasAddBiasAddquantile_3/MatMul:product:0)quantile_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_2/MatMul/ReadVariableOpReadVariableOp)quantile_2_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_2/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_2/BiasAdd/ReadVariableOpReadVariableOp*quantile_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_2/BiasAddBiasAddquantile_2/MatMul:product:0)quantile_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_1/MatMul/ReadVariableOpReadVariableOp)quantile_1_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_1/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_1/BiasAdd/ReadVariableOpReadVariableOp*quantile_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_1/BiasAddBiasAddquantile_1/MatMul:product:0)quantile_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
 quantile_0/MatMul/ReadVariableOpReadVariableOp)quantile_0_matmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0
quantile_0/MatMulMatMul!dropout/dropout/SelectV2:output:0(quantile_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
!quantile_0/BiasAdd/ReadVariableOpReadVariableOp*quantile_0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
quantile_0/BiasAddBiasAddquantile_0/MatMul:product:0)quantile_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџj
IdentityIdentityquantile_0/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_1Identityquantile_1/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_2Identityquantile_2/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_3Identityquantile_3/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_4Identityquantile_4/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_5Identityquantile_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_6Identityquantile_6/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_7Identityquantile_7/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџl

Identity_8Identityquantile_8/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџй
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^embedding/embedding_lookup^embedding_1/embedding_lookup^embedding_2/embedding_lookup^embedding_3/embedding_lookup^embedding_4/embedding_lookup^embedding_5/embedding_lookup^embedding_6/embedding_lookup"^quantile_0/BiasAdd/ReadVariableOp!^quantile_0/MatMul/ReadVariableOp"^quantile_1/BiasAdd/ReadVariableOp!^quantile_1/MatMul/ReadVariableOp"^quantile_2/BiasAdd/ReadVariableOp!^quantile_2/MatMul/ReadVariableOp"^quantile_3/BiasAdd/ReadVariableOp!^quantile_3/MatMul/ReadVariableOp"^quantile_4/BiasAdd/ReadVariableOp!^quantile_4/MatMul/ReadVariableOp"^quantile_5/BiasAdd/ReadVariableOp!^quantile_5/MatMul/ReadVariableOp"^quantile_6/BiasAdd/ReadVariableOp!^quantile_6/MatMul/ReadVariableOp"^quantile_7/BiasAdd/ReadVariableOp!^quantile_7/MatMul/ReadVariableOp"^quantile_8/BiasAdd/ReadVariableOp!^quantile_8/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup2<
embedding_1/embedding_lookupembedding_1/embedding_lookup2<
embedding_2/embedding_lookupembedding_2/embedding_lookup2<
embedding_3/embedding_lookupembedding_3/embedding_lookup2<
embedding_4/embedding_lookupembedding_4/embedding_lookup2<
embedding_5/embedding_lookupembedding_5/embedding_lookup2<
embedding_6/embedding_lookupembedding_6/embedding_lookup2F
!quantile_0/BiasAdd/ReadVariableOp!quantile_0/BiasAdd/ReadVariableOp2D
 quantile_0/MatMul/ReadVariableOp quantile_0/MatMul/ReadVariableOp2F
!quantile_1/BiasAdd/ReadVariableOp!quantile_1/BiasAdd/ReadVariableOp2D
 quantile_1/MatMul/ReadVariableOp quantile_1/MatMul/ReadVariableOp2F
!quantile_2/BiasAdd/ReadVariableOp!quantile_2/BiasAdd/ReadVariableOp2D
 quantile_2/MatMul/ReadVariableOp quantile_2/MatMul/ReadVariableOp2F
!quantile_3/BiasAdd/ReadVariableOp!quantile_3/BiasAdd/ReadVariableOp2D
 quantile_3/MatMul/ReadVariableOp quantile_3/MatMul/ReadVariableOp2F
!quantile_4/BiasAdd/ReadVariableOp!quantile_4/BiasAdd/ReadVariableOp2D
 quantile_4/MatMul/ReadVariableOp quantile_4/MatMul/ReadVariableOp2F
!quantile_5/BiasAdd/ReadVariableOp!quantile_5/BiasAdd/ReadVariableOp2D
 quantile_5/MatMul/ReadVariableOp quantile_5/MatMul/ReadVariableOp2F
!quantile_6/BiasAdd/ReadVariableOp!quantile_6/BiasAdd/ReadVariableOp2D
 quantile_6/MatMul/ReadVariableOp quantile_6/MatMul/ReadVariableOp2F
!quantile_7/BiasAdd/ReadVariableOp!quantile_7/BiasAdd/ReadVariableOp2D
 quantile_7/MatMul/ReadVariableOp quantile_7/MatMul/ReadVariableOp2F
!quantile_8/BiasAdd/ReadVariableOp!quantile_8/BiasAdd/ReadVariableOp2D
 quantile_8/MatMul/ReadVariableOp quantile_8/MatMul/ReadVariableOp:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Ћ
H
,__inference_flatten_4_layer_call_fn_25808733

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_25806960`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Э

-__inference_quantile_6_layer_call_fn_25808956

inputs
unknown:	р]
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_6_layer_call_and_return_conditional_losses_25807066o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_5_layer_call_and_return_conditional_losses_25808667

inputs+
embedding_lookup_25808661:8
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25808661Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25808661*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25808661*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
c
G__inference_flatten_4_layer_call_and_return_conditional_losses_25808739

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_8_layer_call_and_return_conditional_losses_25809004

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_7_layer_call_and_return_conditional_losses_25808985

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Н
a
E__inference_flatten_layer_call_and_return_conditional_losses_25806928

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ

.__inference_embedding_3_layer_call_fn_25808623

inputs
unknown:
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_3_layer_call_and_return_conditional_losses_25806876s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ

.__inference_embedding_1_layer_call_fn_25808589

inputs
unknown:
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_25806904s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
П
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_25806976

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_5_layer_call_and_return_conditional_losses_25808947

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Я	
њ
H__inference_quantile_3_layer_call_and_return_conditional_losses_25808909

inputs1
matmul_readvariableop_resource:	р]-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	р]*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:џџџџџџџџџр]: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:џџџџџџџџџр]
 
_user_specified_nameinputs
Њ
Ў
.__inference_concatenate_layer_call_fn_25808773
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identity
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_25806991`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
П
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_25808761

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџX
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ї
F
*__inference_flatten_layer_call_fn_25808689

inputs
identityА
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_25806928`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ђ

і
C__inference_dense_layer_call_and_return_conditional_losses_25808806

inputs1
matmul_readvariableop_resource:	1р].
biasadd_readvariableop_resource:	р]
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	1р]*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:р]*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџр]Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџр]b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:џџџџџџџџџр]w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:џџџџџџџџџ1
 
_user_specified_nameinputs
Л
Щ
I__inference_concatenate_layer_call_and_return_conditional_losses_25808786
inputs_0
inputs_1
inputs_2
inputs_3
inputs_4
inputs_5
inputs_6
inputs_7
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Г
concatConcatV2inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ1W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ1"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*­
_input_shapes
:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_7:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_6:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_5:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_4:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_3:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0

Ў
C__inference_model_layer_call_and_return_conditional_losses_25807281
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8&
embedding_6_25807187:&
embedding_5_25807190:8&
embedding_4_25807193:&
embedding_3_25807196:&
embedding_2_25807199:&
embedding_1_25807202:$
embedding_25807205:!
dense_25807216:	1р]
dense_25807218:	р]&
quantile_8_25807227:	р]!
quantile_8_25807229:&
quantile_7_25807232:	р]!
quantile_7_25807234:&
quantile_6_25807237:	р]!
quantile_6_25807239:&
quantile_5_25807242:	р]!
quantile_5_25807244:&
quantile_4_25807247:	р]!
quantile_4_25807249:&
quantile_3_25807252:	р]!
quantile_3_25807254:&
quantile_2_25807257:	р]!
quantile_2_25807259:&
quantile_1_25807262:	р]!
quantile_1_25807264:&
quantile_0_25807267:	р]!
quantile_0_25807269:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8Ђdense/StatefulPartitionedCallЂ!embedding/StatefulPartitionedCallЂ#embedding_1/StatefulPartitionedCallЂ#embedding_2/StatefulPartitionedCallЂ#embedding_3/StatefulPartitionedCallЂ#embedding_4/StatefulPartitionedCallЂ#embedding_5/StatefulPartitionedCallЂ#embedding_6/StatefulPartitionedCallЂ"quantile_0/StatefulPartitionedCallЂ"quantile_1/StatefulPartitionedCallЂ"quantile_2/StatefulPartitionedCallЂ"quantile_3/StatefulPartitionedCallЂ"quantile_4/StatefulPartitionedCallЂ"quantile_5/StatefulPartitionedCallЂ"quantile_6/StatefulPartitionedCallЂ"quantile_7/StatefulPartitionedCallЂ"quantile_8/StatefulPartitionedCallя
#embedding_6/StatefulPartitionedCallStatefulPartitionedCallinput_7embedding_6_25807187*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_6_layer_call_and_return_conditional_losses_25806834я
#embedding_5/StatefulPartitionedCallStatefulPartitionedCallinput_6embedding_5_25807190*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_5_layer_call_and_return_conditional_losses_25806848я
#embedding_4/StatefulPartitionedCallStatefulPartitionedCallinput_5embedding_4_25807193*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_4_layer_call_and_return_conditional_losses_25806862я
#embedding_3/StatefulPartitionedCallStatefulPartitionedCallinput_4embedding_3_25807196*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_3_layer_call_and_return_conditional_losses_25806876я
#embedding_2/StatefulPartitionedCallStatefulPartitionedCallinput_3embedding_2_25807199*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_2_layer_call_and_return_conditional_losses_25806890я
#embedding_1/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_1_25807202*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_1_layer_call_and_return_conditional_losses_25806904щ
!embedding/StatefulPartitionedCallStatefulPartitionedCallinput_1embedding_25807205*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_25806918м
flatten/PartitionedCallPartitionedCall*embedding/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_25806928т
flatten_1/PartitionedCallPartitionedCall,embedding_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_1_layer_call_and_return_conditional_losses_25806936т
flatten_2/PartitionedCallPartitionedCall,embedding_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_2_layer_call_and_return_conditional_losses_25806944т
flatten_3/PartitionedCallPartitionedCall,embedding_3/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_25806952т
flatten_4/PartitionedCallPartitionedCall,embedding_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_4_layer_call_and_return_conditional_losses_25806960т
flatten_5/PartitionedCallPartitionedCall,embedding_5/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_5_layer_call_and_return_conditional_losses_25806968т
flatten_6/PartitionedCallPartitionedCall,embedding_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_25806976Т
concatenate/PartitionedCallPartitionedCall flatten/PartitionedCall:output:0"flatten_1/PartitionedCall:output:0"flatten_2/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0"flatten_5/PartitionedCall:output:0"flatten_6/PartitionedCall:output:0input_8*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_25806991
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_25807216dense_25807218*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_25807004й
dropout/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџр]* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *N
fIRG
E__inference_dropout_layer_call_and_return_conditional_losses_25807225
"quantile_8/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_8_25807227quantile_8_25807229*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_8_layer_call_and_return_conditional_losses_25807034
"quantile_7/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_7_25807232quantile_7_25807234*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_7_layer_call_and_return_conditional_losses_25807050
"quantile_6/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_6_25807237quantile_6_25807239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_6_layer_call_and_return_conditional_losses_25807066
"quantile_5/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_5_25807242quantile_5_25807244*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_5_layer_call_and_return_conditional_losses_25807082
"quantile_4/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_4_25807247quantile_4_25807249*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_4_layer_call_and_return_conditional_losses_25807098
"quantile_3/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_3_25807252quantile_3_25807254*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_3_layer_call_and_return_conditional_losses_25807114
"quantile_2/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_2_25807257quantile_2_25807259*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_2_layer_call_and_return_conditional_losses_25807130
"quantile_1/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_1_25807262quantile_1_25807264*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_1_layer_call_and_return_conditional_losses_25807146
"quantile_0/StatefulPartitionedCallStatefulPartitionedCall dropout/PartitionedCall:output:0quantile_0_25807267quantile_0_25807269*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_quantile_0_layer_call_and_return_conditional_losses_25807162z
IdentityIdentity+quantile_0/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_1Identity+quantile_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_2Identity+quantile_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_3Identity+quantile_3/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_4Identity+quantile_4/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_5Identity+quantile_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_6Identity+quantile_6/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_7Identity+quantile_7/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ|

Identity_8Identity+quantile_8/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџЛ
NoOpNoOp^dense/StatefulPartitionedCall"^embedding/StatefulPartitionedCall$^embedding_1/StatefulPartitionedCall$^embedding_2/StatefulPartitionedCall$^embedding_3/StatefulPartitionedCall$^embedding_4/StatefulPartitionedCall$^embedding_5/StatefulPartitionedCall$^embedding_6/StatefulPartitionedCall#^quantile_0/StatefulPartitionedCall#^quantile_1/StatefulPartitionedCall#^quantile_2/StatefulPartitionedCall#^quantile_3/StatefulPartitionedCall#^quantile_4/StatefulPartitionedCall#^quantile_5/StatefulPartitionedCall#^quantile_6/StatefulPartitionedCall#^quantile_7/StatefulPartitionedCall#^quantile_8/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2F
!embedding/StatefulPartitionedCall!embedding/StatefulPartitionedCall2J
#embedding_1/StatefulPartitionedCall#embedding_1/StatefulPartitionedCall2J
#embedding_2/StatefulPartitionedCall#embedding_2/StatefulPartitionedCall2J
#embedding_3/StatefulPartitionedCall#embedding_3/StatefulPartitionedCall2J
#embedding_4/StatefulPartitionedCall#embedding_4/StatefulPartitionedCall2J
#embedding_5/StatefulPartitionedCall#embedding_5/StatefulPartitionedCall2J
#embedding_6/StatefulPartitionedCall#embedding_6/StatefulPartitionedCall2H
"quantile_0/StatefulPartitionedCall"quantile_0/StatefulPartitionedCall2H
"quantile_1/StatefulPartitionedCall"quantile_1/StatefulPartitionedCall2H
"quantile_2/StatefulPartitionedCall"quantile_2/StatefulPartitionedCall2H
"quantile_3/StatefulPartitionedCall"quantile_3/StatefulPartitionedCall2H
"quantile_4/StatefulPartitionedCall"quantile_4/StatefulPartitionedCall2H
"quantile_5/StatefulPartitionedCall"quantile_5/StatefulPartitionedCall2H
"quantile_6/StatefulPartitionedCall"quantile_6/StatefulPartitionedCall2H
"quantile_7/StatefulPartitionedCall"quantile_7/StatefulPartitionedCall2H
"quantile_8/StatefulPartitionedCall"quantile_8/StatefulPartitionedCall:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
­	
Ј
I__inference_embedding_6_layer_call_and_return_conditional_losses_25808684

inputs+
embedding_lookup_25808678:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25808678Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25808678*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25808678*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
­	
Ј
I__inference_embedding_6_layer_call_and_return_conditional_losses_25806834

inputs+
embedding_lookup_25806828:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25806828Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25806828*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25806828*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ќ

.__inference_embedding_5_layer_call_fn_25808657

inputs
unknown:8
identityЂStatefulPartitionedCallе
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *R
fMRK
I__inference_embedding_5_layer_call_and_return_conditional_losses_25806848s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
юа
Ѕ]
!__inference__traced_save_25809660
file_prefix=
+read_disablecopyonread_embedding_embeddings:A
/read_1_disablecopyonread_embedding_1_embeddings:A
/read_2_disablecopyonread_embedding_2_embeddings:A
/read_3_disablecopyonread_embedding_3_embeddings:A
/read_4_disablecopyonread_embedding_4_embeddings:A
/read_5_disablecopyonread_embedding_5_embeddings:8A
/read_6_disablecopyonread_embedding_6_embeddings:8
%read_7_disablecopyonread_dense_kernel:	1р]2
#read_8_disablecopyonread_dense_bias:	р]=
*read_9_disablecopyonread_quantile_0_kernel:	р]7
)read_10_disablecopyonread_quantile_0_bias:>
+read_11_disablecopyonread_quantile_1_kernel:	р]7
)read_12_disablecopyonread_quantile_1_bias:>
+read_13_disablecopyonread_quantile_2_kernel:	р]7
)read_14_disablecopyonread_quantile_2_bias:>
+read_15_disablecopyonread_quantile_3_kernel:	р]7
)read_16_disablecopyonread_quantile_3_bias:>
+read_17_disablecopyonread_quantile_4_kernel:	р]7
)read_18_disablecopyonread_quantile_4_bias:>
+read_19_disablecopyonread_quantile_5_kernel:	р]7
)read_20_disablecopyonread_quantile_5_bias:>
+read_21_disablecopyonread_quantile_6_kernel:	р]7
)read_22_disablecopyonread_quantile_6_bias:>
+read_23_disablecopyonread_quantile_7_kernel:	р]7
)read_24_disablecopyonread_quantile_7_bias:>
+read_25_disablecopyonread_quantile_8_kernel:	р]7
)read_26_disablecopyonread_quantile_8_bias:-
#read_27_disablecopyonread_iteration:	 1
'read_28_disablecopyonread_learning_rate: G
5read_29_disablecopyonread_adam_m_embedding_embeddings:G
5read_30_disablecopyonread_adam_v_embedding_embeddings:I
7read_31_disablecopyonread_adam_m_embedding_1_embeddings:I
7read_32_disablecopyonread_adam_v_embedding_1_embeddings:I
7read_33_disablecopyonread_adam_m_embedding_2_embeddings:I
7read_34_disablecopyonread_adam_v_embedding_2_embeddings:I
7read_35_disablecopyonread_adam_m_embedding_3_embeddings:I
7read_36_disablecopyonread_adam_v_embedding_3_embeddings:I
7read_37_disablecopyonread_adam_m_embedding_4_embeddings:I
7read_38_disablecopyonread_adam_v_embedding_4_embeddings:I
7read_39_disablecopyonread_adam_m_embedding_5_embeddings:8I
7read_40_disablecopyonread_adam_v_embedding_5_embeddings:8I
7read_41_disablecopyonread_adam_m_embedding_6_embeddings:I
7read_42_disablecopyonread_adam_v_embedding_6_embeddings:@
-read_43_disablecopyonread_adam_m_dense_kernel:	1р]@
-read_44_disablecopyonread_adam_v_dense_kernel:	1р]:
+read_45_disablecopyonread_adam_m_dense_bias:	р]:
+read_46_disablecopyonread_adam_v_dense_bias:	р]E
2read_47_disablecopyonread_adam_m_quantile_0_kernel:	р]E
2read_48_disablecopyonread_adam_v_quantile_0_kernel:	р]>
0read_49_disablecopyonread_adam_m_quantile_0_bias:>
0read_50_disablecopyonread_adam_v_quantile_0_bias:E
2read_51_disablecopyonread_adam_m_quantile_1_kernel:	р]E
2read_52_disablecopyonread_adam_v_quantile_1_kernel:	р]>
0read_53_disablecopyonread_adam_m_quantile_1_bias:>
0read_54_disablecopyonread_adam_v_quantile_1_bias:E
2read_55_disablecopyonread_adam_m_quantile_2_kernel:	р]E
2read_56_disablecopyonread_adam_v_quantile_2_kernel:	р]>
0read_57_disablecopyonread_adam_m_quantile_2_bias:>
0read_58_disablecopyonread_adam_v_quantile_2_bias:E
2read_59_disablecopyonread_adam_m_quantile_3_kernel:	р]E
2read_60_disablecopyonread_adam_v_quantile_3_kernel:	р]>
0read_61_disablecopyonread_adam_m_quantile_3_bias:>
0read_62_disablecopyonread_adam_v_quantile_3_bias:E
2read_63_disablecopyonread_adam_m_quantile_4_kernel:	р]E
2read_64_disablecopyonread_adam_v_quantile_4_kernel:	р]>
0read_65_disablecopyonread_adam_m_quantile_4_bias:>
0read_66_disablecopyonread_adam_v_quantile_4_bias:E
2read_67_disablecopyonread_adam_m_quantile_5_kernel:	р]E
2read_68_disablecopyonread_adam_v_quantile_5_kernel:	р]>
0read_69_disablecopyonread_adam_m_quantile_5_bias:>
0read_70_disablecopyonread_adam_v_quantile_5_bias:E
2read_71_disablecopyonread_adam_m_quantile_6_kernel:	р]E
2read_72_disablecopyonread_adam_v_quantile_6_kernel:	р]>
0read_73_disablecopyonread_adam_m_quantile_6_bias:>
0read_74_disablecopyonread_adam_v_quantile_6_bias:E
2read_75_disablecopyonread_adam_m_quantile_7_kernel:	р]E
2read_76_disablecopyonread_adam_v_quantile_7_kernel:	р]>
0read_77_disablecopyonread_adam_m_quantile_7_bias:>
0read_78_disablecopyonread_adam_v_quantile_7_bias:E
2read_79_disablecopyonread_adam_m_quantile_8_kernel:	р]E
2read_80_disablecopyonread_adam_v_quantile_8_kernel:	р]>
0read_81_disablecopyonread_adam_m_quantile_8_bias:>
0read_82_disablecopyonread_adam_v_quantile_8_bias:+
!read_83_disablecopyonread_total_9: +
!read_84_disablecopyonread_count_9: +
!read_85_disablecopyonread_total_8: +
!read_86_disablecopyonread_count_8: +
!read_87_disablecopyonread_total_7: +
!read_88_disablecopyonread_count_7: +
!read_89_disablecopyonread_total_6: +
!read_90_disablecopyonread_count_6: +
!read_91_disablecopyonread_total_5: +
!read_92_disablecopyonread_count_5: +
!read_93_disablecopyonread_total_4: +
!read_94_disablecopyonread_count_4: +
!read_95_disablecopyonread_total_3: +
!read_96_disablecopyonread_count_3: +
!read_97_disablecopyonread_total_2: +
!read_98_disablecopyonread_count_2: +
!read_99_disablecopyonread_total_1: ,
"read_100_disablecopyonread_count_1: *
 read_101_disablecopyonread_total: *
 read_102_disablecopyonread_count: 
savev2_const
identity_207ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_100/DisableCopyOnReadЂRead_100/ReadVariableOpЂRead_101/DisableCopyOnReadЂRead_101/ReadVariableOpЂRead_102/DisableCopyOnReadЂRead_102/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_36/DisableCopyOnReadЂRead_36/ReadVariableOpЂRead_37/DisableCopyOnReadЂRead_37/ReadVariableOpЂRead_38/DisableCopyOnReadЂRead_38/ReadVariableOpЂRead_39/DisableCopyOnReadЂRead_39/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_40/DisableCopyOnReadЂRead_40/ReadVariableOpЂRead_41/DisableCopyOnReadЂRead_41/ReadVariableOpЂRead_42/DisableCopyOnReadЂRead_42/ReadVariableOpЂRead_43/DisableCopyOnReadЂRead_43/ReadVariableOpЂRead_44/DisableCopyOnReadЂRead_44/ReadVariableOpЂRead_45/DisableCopyOnReadЂRead_45/ReadVariableOpЂRead_46/DisableCopyOnReadЂRead_46/ReadVariableOpЂRead_47/DisableCopyOnReadЂRead_47/ReadVariableOpЂRead_48/DisableCopyOnReadЂRead_48/ReadVariableOpЂRead_49/DisableCopyOnReadЂRead_49/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_50/DisableCopyOnReadЂRead_50/ReadVariableOpЂRead_51/DisableCopyOnReadЂRead_51/ReadVariableOpЂRead_52/DisableCopyOnReadЂRead_52/ReadVariableOpЂRead_53/DisableCopyOnReadЂRead_53/ReadVariableOpЂRead_54/DisableCopyOnReadЂRead_54/ReadVariableOpЂRead_55/DisableCopyOnReadЂRead_55/ReadVariableOpЂRead_56/DisableCopyOnReadЂRead_56/ReadVariableOpЂRead_57/DisableCopyOnReadЂRead_57/ReadVariableOpЂRead_58/DisableCopyOnReadЂRead_58/ReadVariableOpЂRead_59/DisableCopyOnReadЂRead_59/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_60/DisableCopyOnReadЂRead_60/ReadVariableOpЂRead_61/DisableCopyOnReadЂRead_61/ReadVariableOpЂRead_62/DisableCopyOnReadЂRead_62/ReadVariableOpЂRead_63/DisableCopyOnReadЂRead_63/ReadVariableOpЂRead_64/DisableCopyOnReadЂRead_64/ReadVariableOpЂRead_65/DisableCopyOnReadЂRead_65/ReadVariableOpЂRead_66/DisableCopyOnReadЂRead_66/ReadVariableOpЂRead_67/DisableCopyOnReadЂRead_67/ReadVariableOpЂRead_68/DisableCopyOnReadЂRead_68/ReadVariableOpЂRead_69/DisableCopyOnReadЂRead_69/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_70/DisableCopyOnReadЂRead_70/ReadVariableOpЂRead_71/DisableCopyOnReadЂRead_71/ReadVariableOpЂRead_72/DisableCopyOnReadЂRead_72/ReadVariableOpЂRead_73/DisableCopyOnReadЂRead_73/ReadVariableOpЂRead_74/DisableCopyOnReadЂRead_74/ReadVariableOpЂRead_75/DisableCopyOnReadЂRead_75/ReadVariableOpЂRead_76/DisableCopyOnReadЂRead_76/ReadVariableOpЂRead_77/DisableCopyOnReadЂRead_77/ReadVariableOpЂRead_78/DisableCopyOnReadЂRead_78/ReadVariableOpЂRead_79/DisableCopyOnReadЂRead_79/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_80/DisableCopyOnReadЂRead_80/ReadVariableOpЂRead_81/DisableCopyOnReadЂRead_81/ReadVariableOpЂRead_82/DisableCopyOnReadЂRead_82/ReadVariableOpЂRead_83/DisableCopyOnReadЂRead_83/ReadVariableOpЂRead_84/DisableCopyOnReadЂRead_84/ReadVariableOpЂRead_85/DisableCopyOnReadЂRead_85/ReadVariableOpЂRead_86/DisableCopyOnReadЂRead_86/ReadVariableOpЂRead_87/DisableCopyOnReadЂRead_87/ReadVariableOpЂRead_88/DisableCopyOnReadЂRead_88/ReadVariableOpЂRead_89/DisableCopyOnReadЂRead_89/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpЂRead_90/DisableCopyOnReadЂRead_90/ReadVariableOpЂRead_91/DisableCopyOnReadЂRead_91/ReadVariableOpЂRead_92/DisableCopyOnReadЂRead_92/ReadVariableOpЂRead_93/DisableCopyOnReadЂRead_93/ReadVariableOpЂRead_94/DisableCopyOnReadЂRead_94/ReadVariableOpЂRead_95/DisableCopyOnReadЂRead_95/ReadVariableOpЂRead_96/DisableCopyOnReadЂRead_96/ReadVariableOpЂRead_97/DisableCopyOnReadЂRead_97/ReadVariableOpЂRead_98/DisableCopyOnReadЂRead_98/ReadVariableOpЂRead_99/DisableCopyOnReadЂRead_99/ReadVariableOpw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: }
Read/DisableCopyOnReadDisableCopyOnRead+read_disablecopyonread_embedding_embeddings"/device:CPU:0*
_output_shapes
 Ї
Read/ReadVariableOpReadVariableOp+read_disablecopyonread_embedding_embeddings^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_1/DisableCopyOnReadDisableCopyOnRead/read_1_disablecopyonread_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 Џ
Read_1/ReadVariableOpReadVariableOp/read_1_disablecopyonread_embedding_1_embeddings^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_2/DisableCopyOnReadDisableCopyOnRead/read_2_disablecopyonread_embedding_2_embeddings"/device:CPU:0*
_output_shapes
 Џ
Read_2/ReadVariableOpReadVariableOp/read_2_disablecopyonread_embedding_2_embeddings^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_3/DisableCopyOnReadDisableCopyOnRead/read_3_disablecopyonread_embedding_3_embeddings"/device:CPU:0*
_output_shapes
 Џ
Read_3/ReadVariableOpReadVariableOp/read_3_disablecopyonread_embedding_3_embeddings^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_4/DisableCopyOnReadDisableCopyOnRead/read_4_disablecopyonread_embedding_4_embeddings"/device:CPU:0*
_output_shapes
 Џ
Read_4/ReadVariableOpReadVariableOp/read_4_disablecopyonread_embedding_4_embeddings^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_5/DisableCopyOnReadDisableCopyOnRead/read_5_disablecopyonread_embedding_5_embeddings"/device:CPU:0*
_output_shapes
 Џ
Read_5/ReadVariableOpReadVariableOp/read_5_disablecopyonread_embedding_5_embeddings^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:8*
dtype0n
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:8e
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes

:8
Read_6/DisableCopyOnReadDisableCopyOnRead/read_6_disablecopyonread_embedding_6_embeddings"/device:CPU:0*
_output_shapes
 Џ
Read_6/ReadVariableOpReadVariableOp/read_6_disablecopyonread_embedding_6_embeddings^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 І
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	1р]*
dtype0o
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	1р]f
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:	1р]w
Read_8/DisableCopyOnReadDisableCopyOnRead#read_8_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
  
Read_8/ReadVariableOpReadVariableOp#read_8_disablecopyonread_dense_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:р]*
dtype0k
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:р]b
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes	
:р]~
Read_9/DisableCopyOnReadDisableCopyOnRead*read_9_disablecopyonread_quantile_0_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read_9/ReadVariableOpReadVariableOp*read_9_disablecopyonread_quantile_0_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0o
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_quantile_0_bias"/device:CPU:0*
_output_shapes
 Ї
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_quantile_0_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_11/DisableCopyOnReadDisableCopyOnRead+read_11_disablecopyonread_quantile_1_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_11/ReadVariableOpReadVariableOp+read_11_disablecopyonread_quantile_1_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_quantile_1_bias"/device:CPU:0*
_output_shapes
 Ї
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_quantile_1_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_13/DisableCopyOnReadDisableCopyOnRead+read_13_disablecopyonread_quantile_2_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_13/ReadVariableOpReadVariableOp+read_13_disablecopyonread_quantile_2_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_quantile_2_bias"/device:CPU:0*
_output_shapes
 Ї
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_quantile_2_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_15/DisableCopyOnReadDisableCopyOnRead+read_15_disablecopyonread_quantile_3_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_15/ReadVariableOpReadVariableOp+read_15_disablecopyonread_quantile_3_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_16/DisableCopyOnReadDisableCopyOnRead)read_16_disablecopyonread_quantile_3_bias"/device:CPU:0*
_output_shapes
 Ї
Read_16/ReadVariableOpReadVariableOp)read_16_disablecopyonread_quantile_3_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_17/DisableCopyOnReadDisableCopyOnRead+read_17_disablecopyonread_quantile_4_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_17/ReadVariableOpReadVariableOp+read_17_disablecopyonread_quantile_4_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_18/DisableCopyOnReadDisableCopyOnRead)read_18_disablecopyonread_quantile_4_bias"/device:CPU:0*
_output_shapes
 Ї
Read_18/ReadVariableOpReadVariableOp)read_18_disablecopyonread_quantile_4_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_19/DisableCopyOnReadDisableCopyOnRead+read_19_disablecopyonread_quantile_5_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_19/ReadVariableOpReadVariableOp+read_19_disablecopyonread_quantile_5_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_20/DisableCopyOnReadDisableCopyOnRead)read_20_disablecopyonread_quantile_5_bias"/device:CPU:0*
_output_shapes
 Ї
Read_20/ReadVariableOpReadVariableOp)read_20_disablecopyonread_quantile_5_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_21/DisableCopyOnReadDisableCopyOnRead+read_21_disablecopyonread_quantile_6_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_21/ReadVariableOpReadVariableOp+read_21_disablecopyonread_quantile_6_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_22/DisableCopyOnReadDisableCopyOnRead)read_22_disablecopyonread_quantile_6_bias"/device:CPU:0*
_output_shapes
 Ї
Read_22/ReadVariableOpReadVariableOp)read_22_disablecopyonread_quantile_6_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_23/DisableCopyOnReadDisableCopyOnRead+read_23_disablecopyonread_quantile_7_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_23/ReadVariableOpReadVariableOp+read_23_disablecopyonread_quantile_7_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_24/DisableCopyOnReadDisableCopyOnRead)read_24_disablecopyonread_quantile_7_bias"/device:CPU:0*
_output_shapes
 Ї
Read_24/ReadVariableOpReadVariableOp)read_24_disablecopyonread_quantile_7_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_25/DisableCopyOnReadDisableCopyOnRead+read_25_disablecopyonread_quantile_8_kernel"/device:CPU:0*
_output_shapes
 Ў
Read_25/ReadVariableOpReadVariableOp+read_25_disablecopyonread_quantile_8_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]~
Read_26/DisableCopyOnReadDisableCopyOnRead)read_26_disablecopyonread_quantile_8_bias"/device:CPU:0*
_output_shapes
 Ї
Read_26/ReadVariableOpReadVariableOp)read_26_disablecopyonread_quantile_8_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:x
Read_27/DisableCopyOnReadDisableCopyOnRead#read_27_disablecopyonread_iteration"/device:CPU:0*
_output_shapes
 
Read_27/ReadVariableOpReadVariableOp#read_27_disablecopyonread_iteration^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0	g
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0	*
_output_shapes
: ]
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0	*
_output_shapes
: |
Read_28/DisableCopyOnReadDisableCopyOnRead'read_28_disablecopyonread_learning_rate"/device:CPU:0*
_output_shapes
 Ё
Read_28/ReadVariableOpReadVariableOp'read_28_disablecopyonread_learning_rate^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0g
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: ]
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes
: 
Read_29/DisableCopyOnReadDisableCopyOnRead5read_29_disablecopyonread_adam_m_embedding_embeddings"/device:CPU:0*
_output_shapes
 З
Read_29/ReadVariableOpReadVariableOp5read_29_disablecopyonread_adam_m_embedding_embeddings^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_30/DisableCopyOnReadDisableCopyOnRead5read_30_disablecopyonread_adam_v_embedding_embeddings"/device:CPU:0*
_output_shapes
 З
Read_30/ReadVariableOpReadVariableOp5read_30_disablecopyonread_adam_v_embedding_embeddings^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_31/DisableCopyOnReadDisableCopyOnRead7read_31_disablecopyonread_adam_m_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_31/ReadVariableOpReadVariableOp7read_31_disablecopyonread_adam_m_embedding_1_embeddings^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_32/DisableCopyOnReadDisableCopyOnRead7read_32_disablecopyonread_adam_v_embedding_1_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_32/ReadVariableOpReadVariableOp7read_32_disablecopyonread_adam_v_embedding_1_embeddings^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_33/DisableCopyOnReadDisableCopyOnRead7read_33_disablecopyonread_adam_m_embedding_2_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_33/ReadVariableOpReadVariableOp7read_33_disablecopyonread_adam_m_embedding_2_embeddings^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_34/DisableCopyOnReadDisableCopyOnRead7read_34_disablecopyonread_adam_v_embedding_2_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_34/ReadVariableOpReadVariableOp7read_34_disablecopyonread_adam_v_embedding_2_embeddings^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_35/DisableCopyOnReadDisableCopyOnRead7read_35_disablecopyonread_adam_m_embedding_3_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_35/ReadVariableOpReadVariableOp7read_35_disablecopyonread_adam_m_embedding_3_embeddings^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_36/DisableCopyOnReadDisableCopyOnRead7read_36_disablecopyonread_adam_v_embedding_3_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_36/ReadVariableOpReadVariableOp7read_36_disablecopyonread_adam_v_embedding_3_embeddings^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_37/DisableCopyOnReadDisableCopyOnRead7read_37_disablecopyonread_adam_m_embedding_4_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_37/ReadVariableOpReadVariableOp7read_37_disablecopyonread_adam_m_embedding_4_embeddings^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_38/DisableCopyOnReadDisableCopyOnRead7read_38_disablecopyonread_adam_v_embedding_4_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_38/ReadVariableOpReadVariableOp7read_38_disablecopyonread_adam_v_embedding_4_embeddings^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_39/DisableCopyOnReadDisableCopyOnRead7read_39_disablecopyonread_adam_m_embedding_5_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_39/ReadVariableOpReadVariableOp7read_39_disablecopyonread_adam_m_embedding_5_embeddings^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:8*
dtype0o
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:8e
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes

:8
Read_40/DisableCopyOnReadDisableCopyOnRead7read_40_disablecopyonread_adam_v_embedding_5_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_40/ReadVariableOpReadVariableOp7read_40_disablecopyonread_adam_v_embedding_5_embeddings^Read_40/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:8*
dtype0o
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:8e
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*
_output_shapes

:8
Read_41/DisableCopyOnReadDisableCopyOnRead7read_41_disablecopyonread_adam_m_embedding_6_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_41/ReadVariableOpReadVariableOp7read_41_disablecopyonread_adam_m_embedding_6_embeddings^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_42/DisableCopyOnReadDisableCopyOnRead7read_42_disablecopyonread_adam_v_embedding_6_embeddings"/device:CPU:0*
_output_shapes
 Й
Read_42/ReadVariableOpReadVariableOp7read_42_disablecopyonread_adam_v_embedding_6_embeddings^Read_42/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:*
dtype0o
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:e
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*
_output_shapes

:
Read_43/DisableCopyOnReadDisableCopyOnRead-read_43_disablecopyonread_adam_m_dense_kernel"/device:CPU:0*
_output_shapes
 А
Read_43/ReadVariableOpReadVariableOp-read_43_disablecopyonread_adam_m_dense_kernel^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	1р]*
dtype0p
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	1р]f
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes
:	1р]
Read_44/DisableCopyOnReadDisableCopyOnRead-read_44_disablecopyonread_adam_v_dense_kernel"/device:CPU:0*
_output_shapes
 А
Read_44/ReadVariableOpReadVariableOp-read_44_disablecopyonread_adam_v_dense_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	1р]*
dtype0p
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	1р]f
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*
_output_shapes
:	1р]
Read_45/DisableCopyOnReadDisableCopyOnRead+read_45_disablecopyonread_adam_m_dense_bias"/device:CPU:0*
_output_shapes
 Њ
Read_45/ReadVariableOpReadVariableOp+read_45_disablecopyonread_adam_m_dense_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:р]*
dtype0l
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:р]b
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes	
:р]
Read_46/DisableCopyOnReadDisableCopyOnRead+read_46_disablecopyonread_adam_v_dense_bias"/device:CPU:0*
_output_shapes
 Њ
Read_46/ReadVariableOpReadVariableOp+read_46_disablecopyonread_adam_v_dense_bias^Read_46/DisableCopyOnRead"/device:CPU:0*
_output_shapes	
:р]*
dtype0l
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes	
:р]b
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*
_output_shapes	
:р]
Read_47/DisableCopyOnReadDisableCopyOnRead2read_47_disablecopyonread_adam_m_quantile_0_kernel"/device:CPU:0*
_output_shapes
 Е
Read_47/ReadVariableOpReadVariableOp2read_47_disablecopyonread_adam_m_quantile_0_kernel^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_48/DisableCopyOnReadDisableCopyOnRead2read_48_disablecopyonread_adam_v_quantile_0_kernel"/device:CPU:0*
_output_shapes
 Е
Read_48/ReadVariableOpReadVariableOp2read_48_disablecopyonread_adam_v_quantile_0_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0p
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]f
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_49/DisableCopyOnReadDisableCopyOnRead0read_49_disablecopyonread_adam_m_quantile_0_bias"/device:CPU:0*
_output_shapes
 Ў
Read_49/ReadVariableOpReadVariableOp0read_49_disablecopyonread_adam_m_quantile_0_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_50/DisableCopyOnReadDisableCopyOnRead0read_50_disablecopyonread_adam_v_quantile_0_bias"/device:CPU:0*
_output_shapes
 Ў
Read_50/ReadVariableOpReadVariableOp0read_50_disablecopyonread_adam_v_quantile_0_bias^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_51/DisableCopyOnReadDisableCopyOnRead2read_51_disablecopyonread_adam_m_quantile_1_kernel"/device:CPU:0*
_output_shapes
 Е
Read_51/ReadVariableOpReadVariableOp2read_51_disablecopyonread_adam_m_quantile_1_kernel^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_52/DisableCopyOnReadDisableCopyOnRead2read_52_disablecopyonread_adam_v_quantile_1_kernel"/device:CPU:0*
_output_shapes
 Е
Read_52/ReadVariableOpReadVariableOp2read_52_disablecopyonread_adam_v_quantile_1_kernel^Read_52/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_104IdentityRead_52/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_105IdentityIdentity_104:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_53/DisableCopyOnReadDisableCopyOnRead0read_53_disablecopyonread_adam_m_quantile_1_bias"/device:CPU:0*
_output_shapes
 Ў
Read_53/ReadVariableOpReadVariableOp0read_53_disablecopyonread_adam_m_quantile_1_bias^Read_53/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_106IdentityRead_53/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_107IdentityIdentity_106:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_54/DisableCopyOnReadDisableCopyOnRead0read_54_disablecopyonread_adam_v_quantile_1_bias"/device:CPU:0*
_output_shapes
 Ў
Read_54/ReadVariableOpReadVariableOp0read_54_disablecopyonread_adam_v_quantile_1_bias^Read_54/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_108IdentityRead_54/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_109IdentityIdentity_108:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_55/DisableCopyOnReadDisableCopyOnRead2read_55_disablecopyonread_adam_m_quantile_2_kernel"/device:CPU:0*
_output_shapes
 Е
Read_55/ReadVariableOpReadVariableOp2read_55_disablecopyonread_adam_m_quantile_2_kernel^Read_55/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_110IdentityRead_55/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_111IdentityIdentity_110:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_56/DisableCopyOnReadDisableCopyOnRead2read_56_disablecopyonread_adam_v_quantile_2_kernel"/device:CPU:0*
_output_shapes
 Е
Read_56/ReadVariableOpReadVariableOp2read_56_disablecopyonread_adam_v_quantile_2_kernel^Read_56/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_112IdentityRead_56/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_113IdentityIdentity_112:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_57/DisableCopyOnReadDisableCopyOnRead0read_57_disablecopyonread_adam_m_quantile_2_bias"/device:CPU:0*
_output_shapes
 Ў
Read_57/ReadVariableOpReadVariableOp0read_57_disablecopyonread_adam_m_quantile_2_bias^Read_57/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_114IdentityRead_57/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_115IdentityIdentity_114:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_58/DisableCopyOnReadDisableCopyOnRead0read_58_disablecopyonread_adam_v_quantile_2_bias"/device:CPU:0*
_output_shapes
 Ў
Read_58/ReadVariableOpReadVariableOp0read_58_disablecopyonread_adam_v_quantile_2_bias^Read_58/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_116IdentityRead_58/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_117IdentityIdentity_116:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_59/DisableCopyOnReadDisableCopyOnRead2read_59_disablecopyonread_adam_m_quantile_3_kernel"/device:CPU:0*
_output_shapes
 Е
Read_59/ReadVariableOpReadVariableOp2read_59_disablecopyonread_adam_m_quantile_3_kernel^Read_59/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_118IdentityRead_59/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_119IdentityIdentity_118:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_60/DisableCopyOnReadDisableCopyOnRead2read_60_disablecopyonread_adam_v_quantile_3_kernel"/device:CPU:0*
_output_shapes
 Е
Read_60/ReadVariableOpReadVariableOp2read_60_disablecopyonread_adam_v_quantile_3_kernel^Read_60/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_120IdentityRead_60/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_121IdentityIdentity_120:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_61/DisableCopyOnReadDisableCopyOnRead0read_61_disablecopyonread_adam_m_quantile_3_bias"/device:CPU:0*
_output_shapes
 Ў
Read_61/ReadVariableOpReadVariableOp0read_61_disablecopyonread_adam_m_quantile_3_bias^Read_61/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_122IdentityRead_61/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_123IdentityIdentity_122:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_62/DisableCopyOnReadDisableCopyOnRead0read_62_disablecopyonread_adam_v_quantile_3_bias"/device:CPU:0*
_output_shapes
 Ў
Read_62/ReadVariableOpReadVariableOp0read_62_disablecopyonread_adam_v_quantile_3_bias^Read_62/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_124IdentityRead_62/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_125IdentityIdentity_124:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_63/DisableCopyOnReadDisableCopyOnRead2read_63_disablecopyonread_adam_m_quantile_4_kernel"/device:CPU:0*
_output_shapes
 Е
Read_63/ReadVariableOpReadVariableOp2read_63_disablecopyonread_adam_m_quantile_4_kernel^Read_63/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_126IdentityRead_63/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_127IdentityIdentity_126:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_64/DisableCopyOnReadDisableCopyOnRead2read_64_disablecopyonread_adam_v_quantile_4_kernel"/device:CPU:0*
_output_shapes
 Е
Read_64/ReadVariableOpReadVariableOp2read_64_disablecopyonread_adam_v_quantile_4_kernel^Read_64/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_128IdentityRead_64/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_129IdentityIdentity_128:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_65/DisableCopyOnReadDisableCopyOnRead0read_65_disablecopyonread_adam_m_quantile_4_bias"/device:CPU:0*
_output_shapes
 Ў
Read_65/ReadVariableOpReadVariableOp0read_65_disablecopyonread_adam_m_quantile_4_bias^Read_65/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_130IdentityRead_65/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_131IdentityIdentity_130:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_66/DisableCopyOnReadDisableCopyOnRead0read_66_disablecopyonread_adam_v_quantile_4_bias"/device:CPU:0*
_output_shapes
 Ў
Read_66/ReadVariableOpReadVariableOp0read_66_disablecopyonread_adam_v_quantile_4_bias^Read_66/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_132IdentityRead_66/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_133IdentityIdentity_132:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_67/DisableCopyOnReadDisableCopyOnRead2read_67_disablecopyonread_adam_m_quantile_5_kernel"/device:CPU:0*
_output_shapes
 Е
Read_67/ReadVariableOpReadVariableOp2read_67_disablecopyonread_adam_m_quantile_5_kernel^Read_67/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_134IdentityRead_67/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_135IdentityIdentity_134:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_68/DisableCopyOnReadDisableCopyOnRead2read_68_disablecopyonread_adam_v_quantile_5_kernel"/device:CPU:0*
_output_shapes
 Е
Read_68/ReadVariableOpReadVariableOp2read_68_disablecopyonread_adam_v_quantile_5_kernel^Read_68/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_136IdentityRead_68/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_137IdentityIdentity_136:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_69/DisableCopyOnReadDisableCopyOnRead0read_69_disablecopyonread_adam_m_quantile_5_bias"/device:CPU:0*
_output_shapes
 Ў
Read_69/ReadVariableOpReadVariableOp0read_69_disablecopyonread_adam_m_quantile_5_bias^Read_69/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_138IdentityRead_69/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_139IdentityIdentity_138:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_70/DisableCopyOnReadDisableCopyOnRead0read_70_disablecopyonread_adam_v_quantile_5_bias"/device:CPU:0*
_output_shapes
 Ў
Read_70/ReadVariableOpReadVariableOp0read_70_disablecopyonread_adam_v_quantile_5_bias^Read_70/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_140IdentityRead_70/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_141IdentityIdentity_140:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_71/DisableCopyOnReadDisableCopyOnRead2read_71_disablecopyonread_adam_m_quantile_6_kernel"/device:CPU:0*
_output_shapes
 Е
Read_71/ReadVariableOpReadVariableOp2read_71_disablecopyonread_adam_m_quantile_6_kernel^Read_71/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_142IdentityRead_71/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_143IdentityIdentity_142:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_72/DisableCopyOnReadDisableCopyOnRead2read_72_disablecopyonread_adam_v_quantile_6_kernel"/device:CPU:0*
_output_shapes
 Е
Read_72/ReadVariableOpReadVariableOp2read_72_disablecopyonread_adam_v_quantile_6_kernel^Read_72/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_144IdentityRead_72/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_145IdentityIdentity_144:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_73/DisableCopyOnReadDisableCopyOnRead0read_73_disablecopyonread_adam_m_quantile_6_bias"/device:CPU:0*
_output_shapes
 Ў
Read_73/ReadVariableOpReadVariableOp0read_73_disablecopyonread_adam_m_quantile_6_bias^Read_73/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_146IdentityRead_73/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_147IdentityIdentity_146:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_74/DisableCopyOnReadDisableCopyOnRead0read_74_disablecopyonread_adam_v_quantile_6_bias"/device:CPU:0*
_output_shapes
 Ў
Read_74/ReadVariableOpReadVariableOp0read_74_disablecopyonread_adam_v_quantile_6_bias^Read_74/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_148IdentityRead_74/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_149IdentityIdentity_148:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_75/DisableCopyOnReadDisableCopyOnRead2read_75_disablecopyonread_adam_m_quantile_7_kernel"/device:CPU:0*
_output_shapes
 Е
Read_75/ReadVariableOpReadVariableOp2read_75_disablecopyonread_adam_m_quantile_7_kernel^Read_75/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_150IdentityRead_75/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_151IdentityIdentity_150:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_76/DisableCopyOnReadDisableCopyOnRead2read_76_disablecopyonread_adam_v_quantile_7_kernel"/device:CPU:0*
_output_shapes
 Е
Read_76/ReadVariableOpReadVariableOp2read_76_disablecopyonread_adam_v_quantile_7_kernel^Read_76/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_152IdentityRead_76/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_153IdentityIdentity_152:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_77/DisableCopyOnReadDisableCopyOnRead0read_77_disablecopyonread_adam_m_quantile_7_bias"/device:CPU:0*
_output_shapes
 Ў
Read_77/ReadVariableOpReadVariableOp0read_77_disablecopyonread_adam_m_quantile_7_bias^Read_77/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_154IdentityRead_77/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_155IdentityIdentity_154:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_78/DisableCopyOnReadDisableCopyOnRead0read_78_disablecopyonread_adam_v_quantile_7_bias"/device:CPU:0*
_output_shapes
 Ў
Read_78/ReadVariableOpReadVariableOp0read_78_disablecopyonread_adam_v_quantile_7_bias^Read_78/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_156IdentityRead_78/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_157IdentityIdentity_156:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_79/DisableCopyOnReadDisableCopyOnRead2read_79_disablecopyonread_adam_m_quantile_8_kernel"/device:CPU:0*
_output_shapes
 Е
Read_79/ReadVariableOpReadVariableOp2read_79_disablecopyonread_adam_m_quantile_8_kernel^Read_79/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_158IdentityRead_79/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_159IdentityIdentity_158:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_80/DisableCopyOnReadDisableCopyOnRead2read_80_disablecopyonread_adam_v_quantile_8_kernel"/device:CPU:0*
_output_shapes
 Е
Read_80/ReadVariableOpReadVariableOp2read_80_disablecopyonread_adam_v_quantile_8_kernel^Read_80/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:	р]*
dtype0q
Identity_160IdentityRead_80/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:	р]h
Identity_161IdentityIdentity_160:output:0"/device:CPU:0*
T0*
_output_shapes
:	р]
Read_81/DisableCopyOnReadDisableCopyOnRead0read_81_disablecopyonread_adam_m_quantile_8_bias"/device:CPU:0*
_output_shapes
 Ў
Read_81/ReadVariableOpReadVariableOp0read_81_disablecopyonread_adam_m_quantile_8_bias^Read_81/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_162IdentityRead_81/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_163IdentityIdentity_162:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_82/DisableCopyOnReadDisableCopyOnRead0read_82_disablecopyonread_adam_v_quantile_8_bias"/device:CPU:0*
_output_shapes
 Ў
Read_82/ReadVariableOpReadVariableOp0read_82_disablecopyonread_adam_v_quantile_8_bias^Read_82/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0l
Identity_164IdentityRead_82/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:c
Identity_165IdentityIdentity_164:output:0"/device:CPU:0*
T0*
_output_shapes
:v
Read_83/DisableCopyOnReadDisableCopyOnRead!read_83_disablecopyonread_total_9"/device:CPU:0*
_output_shapes
 
Read_83/ReadVariableOpReadVariableOp!read_83_disablecopyonread_total_9^Read_83/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_166IdentityRead_83/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_167IdentityIdentity_166:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_84/DisableCopyOnReadDisableCopyOnRead!read_84_disablecopyonread_count_9"/device:CPU:0*
_output_shapes
 
Read_84/ReadVariableOpReadVariableOp!read_84_disablecopyonread_count_9^Read_84/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_168IdentityRead_84/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_169IdentityIdentity_168:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_85/DisableCopyOnReadDisableCopyOnRead!read_85_disablecopyonread_total_8"/device:CPU:0*
_output_shapes
 
Read_85/ReadVariableOpReadVariableOp!read_85_disablecopyonread_total_8^Read_85/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_170IdentityRead_85/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_171IdentityIdentity_170:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_86/DisableCopyOnReadDisableCopyOnRead!read_86_disablecopyonread_count_8"/device:CPU:0*
_output_shapes
 
Read_86/ReadVariableOpReadVariableOp!read_86_disablecopyonread_count_8^Read_86/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_172IdentityRead_86/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_173IdentityIdentity_172:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_87/DisableCopyOnReadDisableCopyOnRead!read_87_disablecopyonread_total_7"/device:CPU:0*
_output_shapes
 
Read_87/ReadVariableOpReadVariableOp!read_87_disablecopyonread_total_7^Read_87/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_174IdentityRead_87/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_175IdentityIdentity_174:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_88/DisableCopyOnReadDisableCopyOnRead!read_88_disablecopyonread_count_7"/device:CPU:0*
_output_shapes
 
Read_88/ReadVariableOpReadVariableOp!read_88_disablecopyonread_count_7^Read_88/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_176IdentityRead_88/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_177IdentityIdentity_176:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_89/DisableCopyOnReadDisableCopyOnRead!read_89_disablecopyonread_total_6"/device:CPU:0*
_output_shapes
 
Read_89/ReadVariableOpReadVariableOp!read_89_disablecopyonread_total_6^Read_89/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_178IdentityRead_89/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_179IdentityIdentity_178:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_90/DisableCopyOnReadDisableCopyOnRead!read_90_disablecopyonread_count_6"/device:CPU:0*
_output_shapes
 
Read_90/ReadVariableOpReadVariableOp!read_90_disablecopyonread_count_6^Read_90/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_180IdentityRead_90/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_181IdentityIdentity_180:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_91/DisableCopyOnReadDisableCopyOnRead!read_91_disablecopyonread_total_5"/device:CPU:0*
_output_shapes
 
Read_91/ReadVariableOpReadVariableOp!read_91_disablecopyonread_total_5^Read_91/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_182IdentityRead_91/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_183IdentityIdentity_182:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_92/DisableCopyOnReadDisableCopyOnRead!read_92_disablecopyonread_count_5"/device:CPU:0*
_output_shapes
 
Read_92/ReadVariableOpReadVariableOp!read_92_disablecopyonread_count_5^Read_92/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_184IdentityRead_92/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_185IdentityIdentity_184:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_93/DisableCopyOnReadDisableCopyOnRead!read_93_disablecopyonread_total_4"/device:CPU:0*
_output_shapes
 
Read_93/ReadVariableOpReadVariableOp!read_93_disablecopyonread_total_4^Read_93/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_186IdentityRead_93/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_187IdentityIdentity_186:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_94/DisableCopyOnReadDisableCopyOnRead!read_94_disablecopyonread_count_4"/device:CPU:0*
_output_shapes
 
Read_94/ReadVariableOpReadVariableOp!read_94_disablecopyonread_count_4^Read_94/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_188IdentityRead_94/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_189IdentityIdentity_188:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_95/DisableCopyOnReadDisableCopyOnRead!read_95_disablecopyonread_total_3"/device:CPU:0*
_output_shapes
 
Read_95/ReadVariableOpReadVariableOp!read_95_disablecopyonread_total_3^Read_95/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_190IdentityRead_95/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_191IdentityIdentity_190:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_96/DisableCopyOnReadDisableCopyOnRead!read_96_disablecopyonread_count_3"/device:CPU:0*
_output_shapes
 
Read_96/ReadVariableOpReadVariableOp!read_96_disablecopyonread_count_3^Read_96/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_192IdentityRead_96/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_193IdentityIdentity_192:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_97/DisableCopyOnReadDisableCopyOnRead!read_97_disablecopyonread_total_2"/device:CPU:0*
_output_shapes
 
Read_97/ReadVariableOpReadVariableOp!read_97_disablecopyonread_total_2^Read_97/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_194IdentityRead_97/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_195IdentityIdentity_194:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_98/DisableCopyOnReadDisableCopyOnRead!read_98_disablecopyonread_count_2"/device:CPU:0*
_output_shapes
 
Read_98/ReadVariableOpReadVariableOp!read_98_disablecopyonread_count_2^Read_98/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_196IdentityRead_98/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_197IdentityIdentity_196:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_99/DisableCopyOnReadDisableCopyOnRead!read_99_disablecopyonread_total_1"/device:CPU:0*
_output_shapes
 
Read_99/ReadVariableOpReadVariableOp!read_99_disablecopyonread_total_1^Read_99/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0h
Identity_198IdentityRead_99/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_199IdentityIdentity_198:output:0"/device:CPU:0*
T0*
_output_shapes
: x
Read_100/DisableCopyOnReadDisableCopyOnRead"read_100_disablecopyonread_count_1"/device:CPU:0*
_output_shapes
 
Read_100/ReadVariableOpReadVariableOp"read_100_disablecopyonread_count_1^Read_100/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_200IdentityRead_100/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_201IdentityIdentity_200:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_101/DisableCopyOnReadDisableCopyOnRead read_101_disablecopyonread_total"/device:CPU:0*
_output_shapes
 
Read_101/ReadVariableOpReadVariableOp read_101_disablecopyonread_total^Read_101/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_202IdentityRead_101/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_203IdentityIdentity_202:output:0"/device:CPU:0*
T0*
_output_shapes
: v
Read_102/DisableCopyOnReadDisableCopyOnRead read_102_disablecopyonread_count"/device:CPU:0*
_output_shapes
 
Read_102/ReadVariableOpReadVariableOp read_102_disablecopyonread_count^Read_102/DisableCopyOnRead"/device:CPU:0*
_output_shapes
: *
dtype0i
Identity_204IdentityRead_102/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
: _
Identity_205IdentityIdentity_204:output:0"/device:CPU:0*
T0*
_output_shapes
: ,
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*П+
valueЕ+BВ+hB:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-2/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-3/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-4/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/embeddings/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-6/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/7/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/8/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/9/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/10/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/11/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/12/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/13/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/14/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/15/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/16/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/17/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/18/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/19/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/20/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/21/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/22/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/23/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/24/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/25/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/26/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/27/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/28/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/29/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/30/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/31/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/32/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/33/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/34/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/35/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/36/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/37/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/38/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/39/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/40/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/41/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/42/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/43/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/44/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/45/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/46/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/47/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/48/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/49/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/50/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/51/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/52/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/53/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/_variables/54/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/3/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/4/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/5/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/6/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/7/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/8/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/9/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*х
valueлBиhB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Щ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0Identity_105:output:0Identity_107:output:0Identity_109:output:0Identity_111:output:0Identity_113:output:0Identity_115:output:0Identity_117:output:0Identity_119:output:0Identity_121:output:0Identity_123:output:0Identity_125:output:0Identity_127:output:0Identity_129:output:0Identity_131:output:0Identity_133:output:0Identity_135:output:0Identity_137:output:0Identity_139:output:0Identity_141:output:0Identity_143:output:0Identity_145:output:0Identity_147:output:0Identity_149:output:0Identity_151:output:0Identity_153:output:0Identity_155:output:0Identity_157:output:0Identity_159:output:0Identity_161:output:0Identity_163:output:0Identity_165:output:0Identity_167:output:0Identity_169:output:0Identity_171:output:0Identity_173:output:0Identity_175:output:0Identity_177:output:0Identity_179:output:0Identity_181:output:0Identity_183:output:0Identity_185:output:0Identity_187:output:0Identity_189:output:0Identity_191:output:0Identity_193:output:0Identity_195:output:0Identity_197:output:0Identity_199:output:0Identity_201:output:0Identity_203:output:0Identity_205:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *v
dtypesl
j2h	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_206Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_207IdentityIdentity_206:output:0^NoOp*
T0*
_output_shapes
: +
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_100/DisableCopyOnRead^Read_100/ReadVariableOp^Read_101/DisableCopyOnRead^Read_101/ReadVariableOp^Read_102/DisableCopyOnRead^Read_102/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_52/DisableCopyOnRead^Read_52/ReadVariableOp^Read_53/DisableCopyOnRead^Read_53/ReadVariableOp^Read_54/DisableCopyOnRead^Read_54/ReadVariableOp^Read_55/DisableCopyOnRead^Read_55/ReadVariableOp^Read_56/DisableCopyOnRead^Read_56/ReadVariableOp^Read_57/DisableCopyOnRead^Read_57/ReadVariableOp^Read_58/DisableCopyOnRead^Read_58/ReadVariableOp^Read_59/DisableCopyOnRead^Read_59/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_60/DisableCopyOnRead^Read_60/ReadVariableOp^Read_61/DisableCopyOnRead^Read_61/ReadVariableOp^Read_62/DisableCopyOnRead^Read_62/ReadVariableOp^Read_63/DisableCopyOnRead^Read_63/ReadVariableOp^Read_64/DisableCopyOnRead^Read_64/ReadVariableOp^Read_65/DisableCopyOnRead^Read_65/ReadVariableOp^Read_66/DisableCopyOnRead^Read_66/ReadVariableOp^Read_67/DisableCopyOnRead^Read_67/ReadVariableOp^Read_68/DisableCopyOnRead^Read_68/ReadVariableOp^Read_69/DisableCopyOnRead^Read_69/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_70/DisableCopyOnRead^Read_70/ReadVariableOp^Read_71/DisableCopyOnRead^Read_71/ReadVariableOp^Read_72/DisableCopyOnRead^Read_72/ReadVariableOp^Read_73/DisableCopyOnRead^Read_73/ReadVariableOp^Read_74/DisableCopyOnRead^Read_74/ReadVariableOp^Read_75/DisableCopyOnRead^Read_75/ReadVariableOp^Read_76/DisableCopyOnRead^Read_76/ReadVariableOp^Read_77/DisableCopyOnRead^Read_77/ReadVariableOp^Read_78/DisableCopyOnRead^Read_78/ReadVariableOp^Read_79/DisableCopyOnRead^Read_79/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_80/DisableCopyOnRead^Read_80/ReadVariableOp^Read_81/DisableCopyOnRead^Read_81/ReadVariableOp^Read_82/DisableCopyOnRead^Read_82/ReadVariableOp^Read_83/DisableCopyOnRead^Read_83/ReadVariableOp^Read_84/DisableCopyOnRead^Read_84/ReadVariableOp^Read_85/DisableCopyOnRead^Read_85/ReadVariableOp^Read_86/DisableCopyOnRead^Read_86/ReadVariableOp^Read_87/DisableCopyOnRead^Read_87/ReadVariableOp^Read_88/DisableCopyOnRead^Read_88/ReadVariableOp^Read_89/DisableCopyOnRead^Read_89/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp^Read_90/DisableCopyOnRead^Read_90/ReadVariableOp^Read_91/DisableCopyOnRead^Read_91/ReadVariableOp^Read_92/DisableCopyOnRead^Read_92/ReadVariableOp^Read_93/DisableCopyOnRead^Read_93/ReadVariableOp^Read_94/DisableCopyOnRead^Read_94/ReadVariableOp^Read_95/DisableCopyOnRead^Read_95/ReadVariableOp^Read_96/DisableCopyOnRead^Read_96/ReadVariableOp^Read_97/DisableCopyOnRead^Read_97/ReadVariableOp^Read_98/DisableCopyOnRead^Read_98/ReadVariableOp^Read_99/DisableCopyOnRead^Read_99/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "%
identity_207Identity_207:output:0*ч
_input_shapesе
в: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp28
Read_100/DisableCopyOnReadRead_100/DisableCopyOnRead22
Read_100/ReadVariableOpRead_100/ReadVariableOp28
Read_101/DisableCopyOnReadRead_101/DisableCopyOnRead22
Read_101/ReadVariableOpRead_101/ReadVariableOp28
Read_102/DisableCopyOnReadRead_102/DisableCopyOnRead22
Read_102/ReadVariableOpRead_102/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp26
Read_52/DisableCopyOnReadRead_52/DisableCopyOnRead20
Read_52/ReadVariableOpRead_52/ReadVariableOp26
Read_53/DisableCopyOnReadRead_53/DisableCopyOnRead20
Read_53/ReadVariableOpRead_53/ReadVariableOp26
Read_54/DisableCopyOnReadRead_54/DisableCopyOnRead20
Read_54/ReadVariableOpRead_54/ReadVariableOp26
Read_55/DisableCopyOnReadRead_55/DisableCopyOnRead20
Read_55/ReadVariableOpRead_55/ReadVariableOp26
Read_56/DisableCopyOnReadRead_56/DisableCopyOnRead20
Read_56/ReadVariableOpRead_56/ReadVariableOp26
Read_57/DisableCopyOnReadRead_57/DisableCopyOnRead20
Read_57/ReadVariableOpRead_57/ReadVariableOp26
Read_58/DisableCopyOnReadRead_58/DisableCopyOnRead20
Read_58/ReadVariableOpRead_58/ReadVariableOp26
Read_59/DisableCopyOnReadRead_59/DisableCopyOnRead20
Read_59/ReadVariableOpRead_59/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp26
Read_60/DisableCopyOnReadRead_60/DisableCopyOnRead20
Read_60/ReadVariableOpRead_60/ReadVariableOp26
Read_61/DisableCopyOnReadRead_61/DisableCopyOnRead20
Read_61/ReadVariableOpRead_61/ReadVariableOp26
Read_62/DisableCopyOnReadRead_62/DisableCopyOnRead20
Read_62/ReadVariableOpRead_62/ReadVariableOp26
Read_63/DisableCopyOnReadRead_63/DisableCopyOnRead20
Read_63/ReadVariableOpRead_63/ReadVariableOp26
Read_64/DisableCopyOnReadRead_64/DisableCopyOnRead20
Read_64/ReadVariableOpRead_64/ReadVariableOp26
Read_65/DisableCopyOnReadRead_65/DisableCopyOnRead20
Read_65/ReadVariableOpRead_65/ReadVariableOp26
Read_66/DisableCopyOnReadRead_66/DisableCopyOnRead20
Read_66/ReadVariableOpRead_66/ReadVariableOp26
Read_67/DisableCopyOnReadRead_67/DisableCopyOnRead20
Read_67/ReadVariableOpRead_67/ReadVariableOp26
Read_68/DisableCopyOnReadRead_68/DisableCopyOnRead20
Read_68/ReadVariableOpRead_68/ReadVariableOp26
Read_69/DisableCopyOnReadRead_69/DisableCopyOnRead20
Read_69/ReadVariableOpRead_69/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp26
Read_70/DisableCopyOnReadRead_70/DisableCopyOnRead20
Read_70/ReadVariableOpRead_70/ReadVariableOp26
Read_71/DisableCopyOnReadRead_71/DisableCopyOnRead20
Read_71/ReadVariableOpRead_71/ReadVariableOp26
Read_72/DisableCopyOnReadRead_72/DisableCopyOnRead20
Read_72/ReadVariableOpRead_72/ReadVariableOp26
Read_73/DisableCopyOnReadRead_73/DisableCopyOnRead20
Read_73/ReadVariableOpRead_73/ReadVariableOp26
Read_74/DisableCopyOnReadRead_74/DisableCopyOnRead20
Read_74/ReadVariableOpRead_74/ReadVariableOp26
Read_75/DisableCopyOnReadRead_75/DisableCopyOnRead20
Read_75/ReadVariableOpRead_75/ReadVariableOp26
Read_76/DisableCopyOnReadRead_76/DisableCopyOnRead20
Read_76/ReadVariableOpRead_76/ReadVariableOp26
Read_77/DisableCopyOnReadRead_77/DisableCopyOnRead20
Read_77/ReadVariableOpRead_77/ReadVariableOp26
Read_78/DisableCopyOnReadRead_78/DisableCopyOnRead20
Read_78/ReadVariableOpRead_78/ReadVariableOp26
Read_79/DisableCopyOnReadRead_79/DisableCopyOnRead20
Read_79/ReadVariableOpRead_79/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp26
Read_80/DisableCopyOnReadRead_80/DisableCopyOnRead20
Read_80/ReadVariableOpRead_80/ReadVariableOp26
Read_81/DisableCopyOnReadRead_81/DisableCopyOnRead20
Read_81/ReadVariableOpRead_81/ReadVariableOp26
Read_82/DisableCopyOnReadRead_82/DisableCopyOnRead20
Read_82/ReadVariableOpRead_82/ReadVariableOp26
Read_83/DisableCopyOnReadRead_83/DisableCopyOnRead20
Read_83/ReadVariableOpRead_83/ReadVariableOp26
Read_84/DisableCopyOnReadRead_84/DisableCopyOnRead20
Read_84/ReadVariableOpRead_84/ReadVariableOp26
Read_85/DisableCopyOnReadRead_85/DisableCopyOnRead20
Read_85/ReadVariableOpRead_85/ReadVariableOp26
Read_86/DisableCopyOnReadRead_86/DisableCopyOnRead20
Read_86/ReadVariableOpRead_86/ReadVariableOp26
Read_87/DisableCopyOnReadRead_87/DisableCopyOnRead20
Read_87/ReadVariableOpRead_87/ReadVariableOp26
Read_88/DisableCopyOnReadRead_88/DisableCopyOnRead20
Read_88/ReadVariableOpRead_88/ReadVariableOp26
Read_89/DisableCopyOnReadRead_89/DisableCopyOnRead20
Read_89/ReadVariableOpRead_89/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp26
Read_90/DisableCopyOnReadRead_90/DisableCopyOnRead20
Read_90/ReadVariableOpRead_90/ReadVariableOp26
Read_91/DisableCopyOnReadRead_91/DisableCopyOnRead20
Read_91/ReadVariableOpRead_91/ReadVariableOp26
Read_92/DisableCopyOnReadRead_92/DisableCopyOnRead20
Read_92/ReadVariableOpRead_92/ReadVariableOp26
Read_93/DisableCopyOnReadRead_93/DisableCopyOnRead20
Read_93/ReadVariableOpRead_93/ReadVariableOp26
Read_94/DisableCopyOnReadRead_94/DisableCopyOnRead20
Read_94/ReadVariableOpRead_94/ReadVariableOp26
Read_95/DisableCopyOnReadRead_95/DisableCopyOnRead20
Read_95/ReadVariableOpRead_95/ReadVariableOp26
Read_96/DisableCopyOnReadRead_96/DisableCopyOnRead20
Read_96/ReadVariableOpRead_96/ReadVariableOp26
Read_97/DisableCopyOnReadRead_97/DisableCopyOnRead20
Read_97/ReadVariableOpRead_97/ReadVariableOp26
Read_98/DisableCopyOnReadRead_98/DisableCopyOnRead20
Read_98/ReadVariableOpRead_98/ReadVariableOp26
Read_99/DisableCopyOnReadRead_99/DisableCopyOnRead20
Read_99/ReadVariableOpRead_99/ReadVariableOp:h

_output_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
­	
Ј
I__inference_embedding_4_layer_call_and_return_conditional_losses_25806862

inputs+
embedding_lookup_25806856:
identityЂembedding_lookupU
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:џџџџџџџџџП
embedding_lookupResourceGatherembedding_lookup_25806856Cast:y:0*
Tindices0*,
_class"
 loc:@embedding_lookup/25806856*+
_output_shapes
:џџџџџџџџџ*
dtype0Є
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*,
_class"
 loc:@embedding_lookup/25806856*+
_output_shapes
:џџџџџџџџџ
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:џџџџџџџџџw
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџY
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
а"
Н
(__inference_model_layer_call_fn_25807463
input_1
input_2
input_3
input_4
input_5
input_6
input_7
input_8
unknown:
	unknown_0:8
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5:
	unknown_6:	1р]
	unknown_7:	р]
	unknown_8:	р]
	unknown_9:

unknown_10:	р]

unknown_11:

unknown_12:	р]

unknown_13:

unknown_14:	р]

unknown_15:

unknown_16:	р]

unknown_17:

unknown_18:	р]

unknown_19:

unknown_20:	р]

unknown_21:

unknown_22:	р]

unknown_23:

unknown_24:	р]

unknown_25:
identity

identity_1

identity_2

identity_3

identity_4

identity_5

identity_6

identity_7

identity_8ЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2input_3input_4input_5input_6input_7input_8unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*.
Tin'
%2#*
Tout
2	*
_collective_manager_ids
 *С
_output_shapesЎ
Ћ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*=
_read_only_resource_inputs
	
 !"*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_25807390o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_3Identity StatefulPartitionedCall:output:3^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_4Identity StatefulPartitionedCall:output:4^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_5Identity StatefulPartitionedCall:output:5^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_6Identity StatefulPartitionedCall:output:6^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_7Identity StatefulPartitionedCall:output:7^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_8Identity StatefulPartitionedCall:output:8^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"!

identity_6Identity_6:output:0"!

identity_7Identity_7:output:0"!

identity_8Identity_8:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*у
_input_shapesб
Ю:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_8:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_7:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_6:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_5:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_4:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_3:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_2:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	input_1
Ј

,__inference_embedding_layer_call_fn_25808572

inputs
unknown:
identityЂStatefulPartitionedCallг
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *P
fKRI
G__inference_embedding_layer_call_and_return_conditional_losses_25806918s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:џџџџџџџџџ: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs"ѓ
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*и
serving_defaultФ
;
input_10
serving_default_input_1:0џџџџџџџџџ
;
input_20
serving_default_input_2:0џџџџџџџџџ
;
input_30
serving_default_input_3:0џџџџџџџџџ
;
input_40
serving_default_input_4:0џџџџџџџџџ
;
input_50
serving_default_input_5:0џџџџџџџџџ
;
input_60
serving_default_input_6:0џџџџџџџџџ
;
input_70
serving_default_input_7:0џџџџџџџџџ
;
input_80
serving_default_input_8:0џџџџџџџџџ>

quantile_00
StatefulPartitionedCall:0џџџџџџџџџ>

quantile_10
StatefulPartitionedCall:1џџџџџџџџџ>

quantile_20
StatefulPartitionedCall:2џџџџџџџџџ>

quantile_30
StatefulPartitionedCall:3џџџџџџџџџ>

quantile_40
StatefulPartitionedCall:4џџџџџџџџџ>

quantile_50
StatefulPartitionedCall:5џџџџџџџџџ>

quantile_60
StatefulPartitionedCall:6џџџџџџџџџ>

quantile_70
StatefulPartitionedCall:7џџџџџџџџџ>

quantile_80
StatefulPartitionedCall:8џџџџџџџџџtensorflow/serving/predict:Єн
	
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer_with_weights-0
layer-7
	layer_with_weights-1
	layer-8

layer_with_weights-2

layer-9
layer_with_weights-3
layer-10
layer_with_weights-4
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer-15
layer-16
layer-17
layer-18
layer-19
layer-20
layer-21
layer-22
layer_with_weights-7
layer-23
layer-24
layer_with_weights-8
layer-25
layer_with_weights-9
layer-26
layer_with_weights-10
layer-27
layer_with_weights-11
layer-28
layer_with_weights-12
layer-29
layer_with_weights-13
layer-30
 layer_with_weights-14
 layer-31
!layer_with_weights-15
!layer-32
"layer_with_weights-16
"layer-33
#	variables
$trainable_variables
%regularization_losses
&	keras_api
'__call__
*(&call_and_return_all_conditional_losses
)_default_save_signature
*	optimizer
+loss
,
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Е
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses
3
embeddings"
_tf_keras_layer
Е
4	variables
5trainable_variables
6regularization_losses
7	keras_api
8__call__
*9&call_and_return_all_conditional_losses
:
embeddings"
_tf_keras_layer
Е
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
A
embeddings"
_tf_keras_layer
Е
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
F__call__
*G&call_and_return_all_conditional_losses
H
embeddings"
_tf_keras_layer
Е
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses
O
embeddings"
_tf_keras_layer
Е
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
T__call__
*U&call_and_return_all_conditional_losses
V
embeddings"
_tf_keras_layer
Е
W	variables
Xtrainable_variables
Yregularization_losses
Z	keras_api
[__call__
*\&call_and_return_all_conditional_losses
]
embeddings"
_tf_keras_layer
Ѕ
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
d	variables
etrainable_variables
fregularization_losses
g	keras_api
h__call__
*i&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
j	variables
ktrainable_variables
lregularization_losses
m	keras_api
n__call__
*o&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
t__call__
*u&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
z__call__
*{&call_and_return_all_conditional_losses"
_tf_keras_layer
Ї
|	variables
}trainable_variables
~regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
 	keras_api
Ё__call__
+Ђ&call_and_return_all_conditional_losses
Ѓkernel
	Єbias"
_tf_keras_layer
У
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses
Ћkernel
	Ќbias"
_tf_keras_layer
У
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses
Гkernel
	Дbias"
_tf_keras_layer
У
Е	variables
Жtrainable_variables
Зregularization_losses
И	keras_api
Й__call__
+К&call_and_return_all_conditional_losses
Лkernel
	Мbias"
_tf_keras_layer
У
Н	variables
Оtrainable_variables
Пregularization_losses
Р	keras_api
С__call__
+Т&call_and_return_all_conditional_losses
Уkernel
	Фbias"
_tf_keras_layer
У
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
Ыkernel
	Ьbias"
_tf_keras_layer
У
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses
гkernel
	дbias"
_tf_keras_layer
У
е	variables
жtrainable_variables
зregularization_losses
и	keras_api
й__call__
+к&call_and_return_all_conditional_losses
лkernel
	мbias"
_tf_keras_layer
У
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses
уkernel
	фbias"
_tf_keras_layer

30
:1
A2
H3
O4
V5
]6
7
8
Ѓ9
Є10
Ћ11
Ќ12
Г13
Д14
Л15
М16
У17
Ф18
Ы19
Ь20
г21
д22
л23
м24
у25
ф26"
trackable_list_wrapper

30
:1
A2
H3
O4
V5
]6
7
8
Ѓ9
Є10
Ћ11
Ќ12
Г13
Д14
Л15
М16
У17
Ф18
Ы19
Ь20
г21
д22
л23
м24
у25
ф26"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
хnon_trainable_variables
цlayers
чmetrics
 шlayer_regularization_losses
щlayer_metrics
#	variables
$trainable_variables
%regularization_losses
'__call__
)_default_save_signature
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
г
ъtrace_0
ыtrace_1
ьtrace_2
эtrace_32р
(__inference_model_layer_call_fn_25807463
(__inference_model_layer_call_fn_25807644
(__inference_model_layer_call_fn_25808198
(__inference_model_layer_call_fn_25808280Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zъtrace_0zыtrace_1zьtrace_2zэtrace_3
П
юtrace_0
яtrace_1
№trace_2
ёtrace_32Ь
C__inference_model_layer_call_and_return_conditional_losses_25807177
C__inference_model_layer_call_and_return_conditional_losses_25807281
C__inference_model_layer_call_and_return_conditional_losses_25808426
C__inference_model_layer_call_and_return_conditional_losses_25808565Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zюtrace_0zяtrace_1z№trace_2zёtrace_3
B
#__inference__wrapped_model_25806813input_1input_2input_3input_4input_5input_6input_7input_8"
В
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ѓ
ђ
_variables
ѓ_iterations
є_learning_rate
ѕ_index_dict
і
_momentums
ї_velocities
ј_update_step_xla"
experimentalOptimizer
 "
trackable_dict_wrapper
-
љserving_default"
signature_map
'
30"
trackable_list_wrapper
'
30"
trackable_list_wrapper
 "
trackable_list_wrapper
В
њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ш
џtrace_02Щ
,__inference_embedding_layer_call_fn_25808572
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0

trace_02ф
G__inference_embedding_layer_call_and_return_conditional_losses_25808582
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
&:$2embedding/embeddings
'
:0"
trackable_list_wrapper
'
:0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
4	variables
5trainable_variables
6regularization_losses
8__call__
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_embedding_1_layer_call_fn_25808589
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ц
I__inference_embedding_1_layer_call_and_return_conditional_losses_25808599
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
(:&2embedding_1/embeddings
'
A0"
trackable_list_wrapper
'
A0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_embedding_2_layer_call_fn_25808606
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ц
I__inference_embedding_2_layer_call_and_return_conditional_losses_25808616
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
(:&2embedding_2/embeddings
'
H0"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
B	variables
Ctrainable_variables
Dregularization_losses
F__call__
*G&call_and_return_all_conditional_losses
&G"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_embedding_3_layer_call_fn_25808623
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ц
I__inference_embedding_3_layer_call_and_return_conditional_losses_25808633
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
(:&2embedding_3/embeddings
'
O0"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_embedding_4_layer_call_fn_25808640
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02ц
I__inference_embedding_4_layer_call_and_return_conditional_losses_25808650
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
(:&2embedding_4/embeddings
'
V0"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
P	variables
Qtrainable_variables
Rregularization_losses
T__call__
*U&call_and_return_all_conditional_losses
&U"call_and_return_conditional_losses"
_generic_user_object
ъ
Ђtrace_02Ы
.__inference_embedding_5_layer_call_fn_25808657
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЂtrace_0

Ѓtrace_02ц
I__inference_embedding_5_layer_call_and_return_conditional_losses_25808667
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0
(:&82embedding_5/embeddings
'
]0"
trackable_list_wrapper
'
]0"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
W	variables
Xtrainable_variables
Yregularization_losses
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
ъ
Љtrace_02Ы
.__inference_embedding_6_layer_call_fn_25808674
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЉtrace_0

Њtrace_02ц
I__inference_embedding_6_layer_call_and_return_conditional_losses_25808684
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0
(:&2embedding_6/embeddings
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ц
Аtrace_02Ч
*__inference_flatten_layer_call_fn_25808689
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zАtrace_0

Бtrace_02т
E__inference_flatten_layer_call_and_return_conditional_losses_25808695
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Вnon_trainable_variables
Гlayers
Дmetrics
 Еlayer_regularization_losses
Жlayer_metrics
d	variables
etrainable_variables
fregularization_losses
h__call__
*i&call_and_return_all_conditional_losses
&i"call_and_return_conditional_losses"
_generic_user_object
ш
Зtrace_02Щ
,__inference_flatten_1_layer_call_fn_25808700
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЗtrace_0

Иtrace_02ф
G__inference_flatten_1_layer_call_and_return_conditional_losses_25808706
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zИtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Йnon_trainable_variables
Кlayers
Лmetrics
 Мlayer_regularization_losses
Нlayer_metrics
j	variables
ktrainable_variables
lregularization_losses
n__call__
*o&call_and_return_all_conditional_losses
&o"call_and_return_conditional_losses"
_generic_user_object
ш
Оtrace_02Щ
,__inference_flatten_2_layer_call_fn_25808711
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zОtrace_0

Пtrace_02ф
G__inference_flatten_2_layer_call_and_return_conditional_losses_25808717
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zПtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Рnon_trainable_variables
Сlayers
Тmetrics
 Уlayer_regularization_losses
Фlayer_metrics
p	variables
qtrainable_variables
rregularization_losses
t__call__
*u&call_and_return_all_conditional_losses
&u"call_and_return_conditional_losses"
_generic_user_object
ш
Хtrace_02Щ
,__inference_flatten_3_layer_call_fn_25808722
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zХtrace_0

Цtrace_02ф
G__inference_flatten_3_layer_call_and_return_conditional_losses_25808728
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЦtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
v	variables
wtrainable_variables
xregularization_losses
z__call__
*{&call_and_return_all_conditional_losses
&{"call_and_return_conditional_losses"
_generic_user_object
ш
Ьtrace_02Щ
,__inference_flatten_4_layer_call_fn_25808733
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЬtrace_0

Эtrace_02ф
G__inference_flatten_4_layer_call_and_return_conditional_losses_25808739
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЭtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Е
Юnon_trainable_variables
Яlayers
аmetrics
 бlayer_regularization_losses
вlayer_metrics
|	variables
}trainable_variables
~regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ш
гtrace_02Щ
,__inference_flatten_5_layer_call_fn_25808744
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zгtrace_0

дtrace_02ф
G__inference_flatten_5_layer_call_and_return_conditional_losses_25808750
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zдtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ш
кtrace_02Щ
,__inference_flatten_6_layer_call_fn_25808755
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zкtrace_0

лtrace_02ф
G__inference_flatten_6_layer_call_and_return_conditional_losses_25808761
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zлtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
рlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ъ
сtrace_02Ы
.__inference_concatenate_layer_call_fn_25808773
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zсtrace_0

тtrace_02ц
I__inference_concatenate_layer_call_and_return_conditional_losses_25808786
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zтtrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ф
шtrace_02Х
(__inference_dense_layer_call_fn_25808795
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zшtrace_0
џ
щtrace_02р
C__inference_dense_layer_call_and_return_conditional_losses_25808806
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zщtrace_0
:	1р]2dense/kernel
:р]2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
ъnon_trainable_variables
ыlayers
ьmetrics
 эlayer_regularization_losses
юlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
П
яtrace_0
№trace_12
*__inference_dropout_layer_call_fn_25808811
*__inference_dropout_layer_call_fn_25808816Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zяtrace_0z№trace_1
ѕ
ёtrace_0
ђtrace_12К
E__inference_dropout_layer_call_and_return_conditional_losses_25808828
E__inference_dropout_layer_call_and_return_conditional_losses_25808833Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zёtrace_0zђtrace_1
"
_generic_user_object
0
Ѓ0
Є1"
trackable_list_wrapper
0
Ѓ0
Є1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ѓnon_trainable_variables
єlayers
ѕmetrics
 іlayer_regularization_losses
їlayer_metrics
	variables
trainable_variables
regularization_losses
Ё__call__
+Ђ&call_and_return_all_conditional_losses
'Ђ"call_and_return_conditional_losses"
_generic_user_object
щ
јtrace_02Ъ
-__inference_quantile_0_layer_call_fn_25808842
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zјtrace_0

љtrace_02х
H__inference_quantile_0_layer_call_and_return_conditional_losses_25808852
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zљtrace_0
$:"	р]2quantile_0/kernel
:2quantile_0/bias
0
Ћ0
Ќ1"
trackable_list_wrapper
0
Ћ0
Ќ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
њnon_trainable_variables
ћlayers
ќmetrics
 §layer_regularization_losses
ўlayer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
щ
џtrace_02Ъ
-__inference_quantile_1_layer_call_fn_25808861
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zџtrace_0

trace_02х
H__inference_quantile_1_layer_call_and_return_conditional_losses_25808871
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
$:"	р]2quantile_1/kernel
:2quantile_1/bias
0
Г0
Д1"
trackable_list_wrapper
0
Г0
Д1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
щ
trace_02Ъ
-__inference_quantile_2_layer_call_fn_25808880
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02х
H__inference_quantile_2_layer_call_and_return_conditional_losses_25808890
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
$:"	р]2quantile_2/kernel
:2quantile_2/bias
0
Л0
М1"
trackable_list_wrapper
0
Л0
М1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Е	variables
Жtrainable_variables
Зregularization_losses
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
щ
trace_02Ъ
-__inference_quantile_3_layer_call_fn_25808899
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02х
H__inference_quantile_3_layer_call_and_return_conditional_losses_25808909
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
$:"	р]2quantile_3/kernel
:2quantile_3/bias
0
У0
Ф1"
trackable_list_wrapper
0
У0
Ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Н	variables
Оtrainable_variables
Пregularization_losses
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
щ
trace_02Ъ
-__inference_quantile_4_layer_call_fn_25808918
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02х
H__inference_quantile_4_layer_call_and_return_conditional_losses_25808928
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
$:"	р]2quantile_4/kernel
:2quantile_4/bias
0
Ы0
Ь1"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
щ
trace_02Ъ
-__inference_quantile_5_layer_call_fn_25808937
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0

trace_02х
H__inference_quantile_5_layer_call_and_return_conditional_losses_25808947
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0
$:"	р]2quantile_5/kernel
:2quantile_5/bias
0
г0
д1"
trackable_list_wrapper
0
г0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
  layer_regularization_losses
Ёlayer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
щ
Ђtrace_02Ъ
-__inference_quantile_6_layer_call_fn_25808956
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЂtrace_0

Ѓtrace_02х
H__inference_quantile_6_layer_call_and_return_conditional_losses_25808966
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЃtrace_0
$:"	р]2quantile_6/kernel
:2quantile_6/bias
0
л0
м1"
trackable_list_wrapper
0
л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Єnon_trainable_variables
Ѕlayers
Іmetrics
 Їlayer_regularization_losses
Јlayer_metrics
е	variables
жtrainable_variables
зregularization_losses
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
щ
Љtrace_02Ъ
-__inference_quantile_7_layer_call_fn_25808975
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЉtrace_0

Њtrace_02х
H__inference_quantile_7_layer_call_and_return_conditional_losses_25808985
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zЊtrace_0
$:"	р]2quantile_7/kernel
:2quantile_7/bias
0
у0
ф1"
trackable_list_wrapper
0
у0
ф1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ћnon_trainable_variables
Ќlayers
­metrics
 Ўlayer_regularization_losses
Џlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
щ
Аtrace_02Ъ
-__inference_quantile_8_layer_call_fn_25808994
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zАtrace_0

Бtrace_02х
H__inference_quantile_8_layer_call_and_return_conditional_losses_25809004
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zБtrace_0
$:"	р]2quantile_8/kernel
:2quantile_8/bias
 "
trackable_list_wrapper
І
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33"
trackable_list_wrapper
p
В0
Г1
Д2
Е3
Ж4
З5
И6
Й7
К8
Л9"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЏBЌ
(__inference_model_layer_call_fn_25807463input_1input_2input_3input_4input_5input_6input_7input_8"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЏBЌ
(__inference_model_layer_call_fn_25807644input_1input_2input_3input_4input_5input_6input_7input_8"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЗBД
(__inference_model_layer_call_fn_25808198inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЗBД
(__inference_model_layer_call_fn_25808280inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЪBЧ
C__inference_model_layer_call_and_return_conditional_losses_25807177input_1input_2input_3input_4input_5input_6input_7input_8"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЪBЧ
C__inference_model_layer_call_and_return_conditional_losses_25807281input_1input_2input_3input_4input_5input_6input_7input_8"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
вBЯ
C__inference_model_layer_call_and_return_conditional_losses_25808426inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
вBЯ
C__inference_model_layer_call_and_return_conditional_losses_25808565inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7"Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 

ѓ0
М1
Н2
О3
П4
Р5
С6
Т7
У8
Ф9
Х10
Ц11
Ч12
Ш13
Щ14
Ъ15
Ы16
Ь17
Э18
Ю19
Я20
а21
б22
в23
г24
д25
е26
ж27
з28
и29
й30
к31
л32
м33
н34
о35
п36
р37
с38
т39
у40
ф41
х42
ц43
ч44
ш45
щ46
ъ47
ы48
ь49
э50
ю51
я52
№53
ё54"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper

М0
О1
Р2
Т3
Ф4
Ц5
Ш6
Ъ7
Ь8
Ю9
а10
в11
д12
ж13
и14
к15
м16
о17
р18
т19
ф20
ц21
ш22
ъ23
ь24
ю25
№26"
trackable_list_wrapper

Н0
П1
С2
У3
Х4
Ч5
Щ6
Ы7
Э8
Я9
б10
г11
е12
з13
й14
л15
н16
п17
с18
у19
х20
ч21
щ22
ы23
э24
я25
ё26"
trackable_list_wrapper
Е2ВЏ
ІВЂ
FullArgSpec*
args"

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 0
B
&__inference_signature_wrapper_25808116input_1input_2input_3input_4input_5input_6input_7input_8"
В
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_embedding_layer_call_fn_25808572inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_embedding_layer_call_and_return_conditional_losses_25808582inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_embedding_1_layer_call_fn_25808589inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_embedding_1_layer_call_and_return_conditional_losses_25808599inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_embedding_2_layer_call_fn_25808606inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_embedding_2_layer_call_and_return_conditional_losses_25808616inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_embedding_3_layer_call_fn_25808623inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_embedding_3_layer_call_and_return_conditional_losses_25808633inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_embedding_4_layer_call_fn_25808640inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_embedding_4_layer_call_and_return_conditional_losses_25808650inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_embedding_5_layer_call_fn_25808657inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_embedding_5_layer_call_and_return_conditional_losses_25808667inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
иBе
.__inference_embedding_6_layer_call_fn_25808674inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ѓB№
I__inference_embedding_6_layer_call_and_return_conditional_losses_25808684inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
дBб
*__inference_flatten_layer_call_fn_25808689inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
яBь
E__inference_flatten_layer_call_and_return_conditional_losses_25808695inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_flatten_1_layer_call_fn_25808700inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_flatten_1_layer_call_and_return_conditional_losses_25808706inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_flatten_2_layer_call_fn_25808711inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_flatten_2_layer_call_and_return_conditional_losses_25808717inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_flatten_3_layer_call_fn_25808722inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_flatten_3_layer_call_and_return_conditional_losses_25808728inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_flatten_4_layer_call_fn_25808733inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_flatten_4_layer_call_and_return_conditional_losses_25808739inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_flatten_5_layer_call_fn_25808744inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_flatten_5_layer_call_and_return_conditional_losses_25808750inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
жBг
,__inference_flatten_6_layer_call_fn_25808755inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ёBю
G__inference_flatten_6_layer_call_and_return_conditional_losses_25808761inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 B
.__inference_concatenate_layer_call_fn_25808773inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЛBИ
I__inference_concatenate_layer_call_and_return_conditional_losses_25808786inputs_0inputs_1inputs_2inputs_3inputs_4inputs_5inputs_6inputs_7"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
вBЯ
(__inference_dense_layer_call_fn_25808795inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
эBъ
C__inference_dense_layer_call_and_return_conditional_losses_25808806inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
хBт
*__inference_dropout_layer_call_fn_25808811inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
хBт
*__inference_dropout_layer_call_fn_25808816inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
E__inference_dropout_layer_call_and_return_conditional_losses_25808828inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B§
E__inference_dropout_layer_call_and_return_conditional_losses_25808833inputs"Љ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_0_layer_call_fn_25808842inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_0_layer_call_and_return_conditional_losses_25808852inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_1_layer_call_fn_25808861inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_1_layer_call_and_return_conditional_losses_25808871inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_2_layer_call_fn_25808880inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_2_layer_call_and_return_conditional_losses_25808890inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_3_layer_call_fn_25808899inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_3_layer_call_and_return_conditional_losses_25808909inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_4_layer_call_fn_25808918inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_4_layer_call_and_return_conditional_losses_25808928inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_5_layer_call_fn_25808937inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_5_layer_call_and_return_conditional_losses_25808947inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_6_layer_call_fn_25808956inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_6_layer_call_and_return_conditional_losses_25808966inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_7_layer_call_fn_25808975inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_7_layer_call_and_return_conditional_losses_25808985inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
зBд
-__inference_quantile_8_layer_call_fn_25808994inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ђBя
H__inference_quantile_8_layer_call_and_return_conditional_losses_25809004inputs"
В
FullArgSpec
args

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
R
ђ	variables
ѓ	keras_api

єtotal

ѕcount"
_tf_keras_metric
R
і	variables
ї	keras_api

јtotal

љcount"
_tf_keras_metric
R
њ	variables
ћ	keras_api

ќtotal

§count"
_tf_keras_metric
R
ў	variables
џ	keras_api

total

count"
_tf_keras_metric
R
	variables
	keras_api

total

count"
_tf_keras_metric
R
	variables
	keras_api

total

count"
_tf_keras_metric
R
	variables
	keras_api

total

count"
_tf_keras_metric
R
	variables
	keras_api

total

count"
_tf_keras_metric
R
	variables
	keras_api

total

count"
_tf_keras_metric
R
	variables
	keras_api

total

count"
_tf_keras_metric
+:)2Adam/m/embedding/embeddings
+:)2Adam/v/embedding/embeddings
-:+2Adam/m/embedding_1/embeddings
-:+2Adam/v/embedding_1/embeddings
-:+2Adam/m/embedding_2/embeddings
-:+2Adam/v/embedding_2/embeddings
-:+2Adam/m/embedding_3/embeddings
-:+2Adam/v/embedding_3/embeddings
-:+2Adam/m/embedding_4/embeddings
-:+2Adam/v/embedding_4/embeddings
-:+82Adam/m/embedding_5/embeddings
-:+82Adam/v/embedding_5/embeddings
-:+2Adam/m/embedding_6/embeddings
-:+2Adam/v/embedding_6/embeddings
$:"	1р]2Adam/m/dense/kernel
$:"	1р]2Adam/v/dense/kernel
:р]2Adam/m/dense/bias
:р]2Adam/v/dense/bias
):'	р]2Adam/m/quantile_0/kernel
):'	р]2Adam/v/quantile_0/kernel
": 2Adam/m/quantile_0/bias
": 2Adam/v/quantile_0/bias
):'	р]2Adam/m/quantile_1/kernel
):'	р]2Adam/v/quantile_1/kernel
": 2Adam/m/quantile_1/bias
": 2Adam/v/quantile_1/bias
):'	р]2Adam/m/quantile_2/kernel
):'	р]2Adam/v/quantile_2/kernel
": 2Adam/m/quantile_2/bias
": 2Adam/v/quantile_2/bias
):'	р]2Adam/m/quantile_3/kernel
):'	р]2Adam/v/quantile_3/kernel
": 2Adam/m/quantile_3/bias
": 2Adam/v/quantile_3/bias
):'	р]2Adam/m/quantile_4/kernel
):'	р]2Adam/v/quantile_4/kernel
": 2Adam/m/quantile_4/bias
": 2Adam/v/quantile_4/bias
):'	р]2Adam/m/quantile_5/kernel
):'	р]2Adam/v/quantile_5/kernel
": 2Adam/m/quantile_5/bias
": 2Adam/v/quantile_5/bias
):'	р]2Adam/m/quantile_6/kernel
):'	р]2Adam/v/quantile_6/kernel
": 2Adam/m/quantile_6/bias
": 2Adam/v/quantile_6/bias
):'	р]2Adam/m/quantile_7/kernel
):'	р]2Adam/v/quantile_7/kernel
": 2Adam/m/quantile_7/bias
": 2Adam/v/quantile_7/bias
):'	р]2Adam/m/quantile_8/kernel
):'	р]2Adam/v/quantile_8/kernel
": 2Adam/m/quantile_8/bias
": 2Adam/v/quantile_8/bias
0
є0
ѕ1"
trackable_list_wrapper
.
ђ	variables"
_generic_user_object
:  (2total
:  (2count
0
ј0
љ1"
trackable_list_wrapper
.
і	variables"
_generic_user_object
:  (2total
:  (2count
0
ќ0
§1"
trackable_list_wrapper
.
њ	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
ў	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2countц
#__inference__wrapped_model_25806813О/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄЏЂЋ
ЃЂ

!
input_1џџџџџџџџџ
!
input_2џџџџџџџџџ
!
input_3џџџџџџџџџ
!
input_4џџџџџџџџџ
!
input_5џџџџџџџџџ
!
input_6џџџџџџџџџ
!
input_7џџџџџџџџџ
!
input_8џџџџџџџџџ
Њ "иЊд
2

quantile_0$!

quantile_0џџџџџџџџџ
2

quantile_1$!

quantile_1џџџџџџџџџ
2

quantile_2$!

quantile_2џџџџџџџџџ
2

quantile_3$!

quantile_3џџџџџџџџџ
2

quantile_4$!

quantile_4џџџџџџџџџ
2

quantile_5$!

quantile_5џџџџџџџџџ
2

quantile_6$!

quantile_6џџџџџџџџџ
2

quantile_7$!

quantile_7џџџџџџџџџ
2

quantile_8$!

quantile_8џџџџџџџџџЖ
I__inference_concatenate_layer_call_and_return_conditional_losses_25808786шЗЂГ
ЋЂЇ
Є 
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
"
inputs_6џџџџџџџџџ
"
inputs_7џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ1
 
.__inference_concatenate_layer_call_fn_25808773нЗЂГ
ЋЂЇ
Є 
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
"
inputs_6џџџџџџџџџ
"
inputs_7џџџџџџџџџ
Њ "!
unknownџџџџџџџџџ1­
C__inference_dense_layer_call_and_return_conditional_losses_25808806f/Ђ,
%Ђ"
 
inputsџџџџџџџџџ1
Њ "-Ђ*
# 
tensor_0џџџџџџџџџр]
 
(__inference_dense_layer_call_fn_25808795[/Ђ,
%Ђ"
 
inputsџџџџџџџџџ1
Њ ""
unknownџџџџџџџџџр]Ў
E__inference_dropout_layer_call_and_return_conditional_losses_25808828e4Ђ1
*Ђ'
!
inputsџџџџџџџџџр]
p
Њ "-Ђ*
# 
tensor_0џџџџџџџџџр]
 Ў
E__inference_dropout_layer_call_and_return_conditional_losses_25808833e4Ђ1
*Ђ'
!
inputsџџџџџџџџџр]
p 
Њ "-Ђ*
# 
tensor_0џџџџџџџџџр]
 
*__inference_dropout_layer_call_fn_25808811Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџр]
p
Њ ""
unknownџџџџџџџџџр]
*__inference_dropout_layer_call_fn_25808816Z4Ђ1
*Ђ'
!
inputsџџџџџџџџџр]
p 
Њ ""
unknownџџџџџџџџџр]Г
I__inference_embedding_1_layer_call_and_return_conditional_losses_25808599f:/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
.__inference_embedding_1_layer_call_fn_25808589[:/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџГ
I__inference_embedding_2_layer_call_and_return_conditional_losses_25808616fA/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
.__inference_embedding_2_layer_call_fn_25808606[A/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџГ
I__inference_embedding_3_layer_call_and_return_conditional_losses_25808633fH/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
.__inference_embedding_3_layer_call_fn_25808623[H/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџГ
I__inference_embedding_4_layer_call_and_return_conditional_losses_25808650fO/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
.__inference_embedding_4_layer_call_fn_25808640[O/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџГ
I__inference_embedding_5_layer_call_and_return_conditional_losses_25808667fV/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
.__inference_embedding_5_layer_call_fn_25808657[V/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџГ
I__inference_embedding_6_layer_call_and_return_conditional_losses_25808684f]/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
.__inference_embedding_6_layer_call_fn_25808674[]/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџБ
G__inference_embedding_layer_call_and_return_conditional_losses_25808582f3/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
,__inference_embedding_layer_call_fn_25808572[3/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџЎ
G__inference_flatten_1_layer_call_and_return_conditional_losses_25808706c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
,__inference_flatten_1_layer_call_fn_25808700X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЎ
G__inference_flatten_2_layer_call_and_return_conditional_losses_25808717c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
,__inference_flatten_2_layer_call_fn_25808711X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЎ
G__inference_flatten_3_layer_call_and_return_conditional_losses_25808728c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
,__inference_flatten_3_layer_call_fn_25808722X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЎ
G__inference_flatten_4_layer_call_and_return_conditional_losses_25808739c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
,__inference_flatten_4_layer_call_fn_25808733X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЎ
G__inference_flatten_5_layer_call_and_return_conditional_losses_25808750c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
,__inference_flatten_5_layer_call_fn_25808744X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЎ
G__inference_flatten_6_layer_call_and_return_conditional_losses_25808761c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
,__inference_flatten_6_layer_call_fn_25808755X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЌ
E__inference_flatten_layer_call_and_return_conditional_losses_25808695c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
*__inference_flatten_layer_call_fn_25808689X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџ
C__inference_model_layer_call_and_return_conditional_losses_25807177д/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄЗЂГ
ЋЂЇ

!
input_1џџџџџџџџџ
!
input_2џџџџџџџџџ
!
input_3џџџџџџџџџ
!
input_4џџџџџџџџџ
!
input_5џџџџџџџџџ
!
input_6џџџџџџџџџ
!
input_7џџџџџџџџџ
!
input_8џџџџџџџџџ
p

 
Њ "цЂт
кж
$!

tensor_0_0џџџџџџџџџ
$!

tensor_0_1џџџџџџџџџ
$!

tensor_0_2џџџџџџџџџ
$!

tensor_0_3џџџџџџџџџ
$!

tensor_0_4џџџџџџџџџ
$!

tensor_0_5џџџџџџџџџ
$!

tensor_0_6џџџџџџџџџ
$!

tensor_0_7џџџџџџџџџ
$!

tensor_0_8џџџџџџџџџ
 
C__inference_model_layer_call_and_return_conditional_losses_25807281д/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄЗЂГ
ЋЂЇ

!
input_1џџџџџџџџџ
!
input_2џџџџџџџџџ
!
input_3џџџџџџџџџ
!
input_4џџџџџџџџџ
!
input_5џџџџџџџџџ
!
input_6џџџџџџџџџ
!
input_7џџџџџџџџџ
!
input_8џџџџџџџџџ
p 

 
Њ "цЂт
кж
$!

tensor_0_0џџџџџџџџџ
$!

tensor_0_1џџџџџџџџџ
$!

tensor_0_2џџџџџџџџџ
$!

tensor_0_3џџџџџџџџџ
$!

tensor_0_4џџџџџџџџџ
$!

tensor_0_5џџџџџџџџџ
$!

tensor_0_6џџџџџџџџџ
$!

tensor_0_7џџџџџџџџџ
$!

tensor_0_8џџџџџџџџџ
 Є
C__inference_model_layer_call_and_return_conditional_losses_25808426м/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄПЂЛ
ГЂЏ
Є 
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
"
inputs_6џџџџџџџџџ
"
inputs_7џџџџџџџџџ
p

 
Њ "цЂт
кж
$!

tensor_0_0џџџџџџџџџ
$!

tensor_0_1џџџџџџџџџ
$!

tensor_0_2џџџџџџџџџ
$!

tensor_0_3џџџџџџџџџ
$!

tensor_0_4џџџџџџџџџ
$!

tensor_0_5џџџџџџџџџ
$!

tensor_0_6џџџџџџџџџ
$!

tensor_0_7џџџџџџџџџ
$!

tensor_0_8џџџџџџџџџ
 Є
C__inference_model_layer_call_and_return_conditional_losses_25808565м/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄПЂЛ
ГЂЏ
Є 
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
"
inputs_6џџџџџџџџџ
"
inputs_7џџџџџџџџџ
p 

 
Њ "цЂт
кж
$!

tensor_0_0џџџџџџџџџ
$!

tensor_0_1џџџџџџџџџ
$!

tensor_0_2џџџџџџџџџ
$!

tensor_0_3џџџџџџџџџ
$!

tensor_0_4џџџџџџџџџ
$!

tensor_0_5џџџџџџџџџ
$!

tensor_0_6џџџџџџџџџ
$!

tensor_0_7џџџџџџџџџ
$!

tensor_0_8џџџџџџџџџ
 у
(__inference_model_layer_call_fn_25807463Ж/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄЗЂГ
ЋЂЇ

!
input_1џџџџџџџџџ
!
input_2џџџџџџџџџ
!
input_3џџџџџџџџџ
!
input_4џџџџџџџџџ
!
input_5џџџџџџџџџ
!
input_6џџџџџџџџџ
!
input_7џџџџџџџџџ
!
input_8џџџџџџџџџ
p

 
Њ "ШФ
"
tensor_0џџџџџџџџџ
"
tensor_1џџџџџџџџџ
"
tensor_2џџџџџџџџџ
"
tensor_3џџџџџџџџџ
"
tensor_4џџџџџџџџџ
"
tensor_5џџџџџџџџџ
"
tensor_6џџџџџџџџџ
"
tensor_7џџџџџџџџџ
"
tensor_8џџџџџџџџџу
(__inference_model_layer_call_fn_25807644Ж/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄЗЂГ
ЋЂЇ

!
input_1џџџџџџџџџ
!
input_2џџџџџџџџџ
!
input_3џџџџџџџџџ
!
input_4џџџџџџџџџ
!
input_5џџџџџџџџџ
!
input_6џџџџџџџџџ
!
input_7џџџџџџџџџ
!
input_8џџџџџџџџџ
p 

 
Њ "ШФ
"
tensor_0џџџџџџџџџ
"
tensor_1џџџџџџџџџ
"
tensor_2џџџџџџџџџ
"
tensor_3џџџџџџџџџ
"
tensor_4џџџџџџџџџ
"
tensor_5џџџџџџџџџ
"
tensor_6џџџџџџџџџ
"
tensor_7џџџџџџџџџ
"
tensor_8џџџџџџџџџы
(__inference_model_layer_call_fn_25808198О/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄПЂЛ
ГЂЏ
Є 
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
"
inputs_6џџџџџџџџџ
"
inputs_7џџџџџџџџџ
p

 
Њ "ШФ
"
tensor_0џџџџџџџџџ
"
tensor_1џџџџџџџџџ
"
tensor_2џџџџџџџџџ
"
tensor_3џџџџџџџџџ
"
tensor_4џџџџџџџџџ
"
tensor_5џџџџџџџџџ
"
tensor_6џџџџџџџџџ
"
tensor_7џџџџџџџџџ
"
tensor_8џџџџџџџџџы
(__inference_model_layer_call_fn_25808280О/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄПЂЛ
ГЂЏ
Є 
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
"
inputs_3џџџџџџџџџ
"
inputs_4џџџџџџџџџ
"
inputs_5џџџџџџџџџ
"
inputs_6џџџџџџџџџ
"
inputs_7џџџџџџџџџ
p 

 
Њ "ШФ
"
tensor_0џџџџџџџџџ
"
tensor_1џџџџџџџџџ
"
tensor_2џџџџџџџџџ
"
tensor_3џџџџџџџџџ
"
tensor_4џџџџџџџџџ
"
tensor_5џџџџџџџџџ
"
tensor_6џџџџџџџџџ
"
tensor_7џџџџџџџџџ
"
tensor_8џџџџџџџџџВ
H__inference_quantile_0_layer_call_and_return_conditional_losses_25808852fЃЄ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_0_layer_call_fn_25808842[ЃЄ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџВ
H__inference_quantile_1_layer_call_and_return_conditional_losses_25808871fЋЌ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_1_layer_call_fn_25808861[ЋЌ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџВ
H__inference_quantile_2_layer_call_and_return_conditional_losses_25808890fГД0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_2_layer_call_fn_25808880[ГД0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџВ
H__inference_quantile_3_layer_call_and_return_conditional_losses_25808909fЛМ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_3_layer_call_fn_25808899[ЛМ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџВ
H__inference_quantile_4_layer_call_and_return_conditional_losses_25808928fУФ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_4_layer_call_fn_25808918[УФ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџВ
H__inference_quantile_5_layer_call_and_return_conditional_losses_25808947fЫЬ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_5_layer_call_fn_25808937[ЫЬ0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџВ
H__inference_quantile_6_layer_call_and_return_conditional_losses_25808966fгд0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_6_layer_call_fn_25808956[гд0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџВ
H__inference_quantile_7_layer_call_and_return_conditional_losses_25808985fлм0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_7_layer_call_fn_25808975[лм0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџВ
H__inference_quantile_8_layer_call_and_return_conditional_losses_25809004fуф0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
-__inference_quantile_8_layer_call_fn_25808994[уф0Ђ-
&Ђ#
!
inputsџџџџџџџџџр]
Њ "!
unknownџџџџџџџџџК
&__inference_signature_wrapper_25808116/]VOHA:3уфлмгдЫЬУФЛМГДЋЌЃЄЂќ
Ђ 
єЊ№
,
input_1!
input_1џџџџџџџџџ
,
input_2!
input_2џџџџџџџџџ
,
input_3!
input_3џџџџџџџџџ
,
input_4!
input_4џџџџџџџџџ
,
input_5!
input_5џџџџџџџџџ
,
input_6!
input_6џџџџџџџџџ
,
input_7!
input_7џџџџџџџџџ
,
input_8!
input_8џџџџџџџџџ"иЊд
2

quantile_0$!

quantile_0џџџџџџџџџ
2

quantile_1$!

quantile_1џџџџџџџџџ
2

quantile_2$!

quantile_2џџџџџџџџџ
2

quantile_3$!

quantile_3џџџџџџџџџ
2

quantile_4$!

quantile_4џџџџџџџџџ
2

quantile_5$!

quantile_5џџџџџџџџџ
2

quantile_6$!

quantile_6џџџџџџџџџ
2

quantile_7$!

quantile_7џџџџџџџџџ
2

quantile_8$!

quantile_8џџџџџџџџџ