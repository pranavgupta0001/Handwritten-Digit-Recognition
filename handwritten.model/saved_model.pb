
Ü
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
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
executor_typestring ¨
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8ç

Adam/softmaxLayer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/softmaxLayer/bias/v

,Adam/softmaxLayer/bias/v/Read/ReadVariableOpReadVariableOpAdam/softmaxLayer/bias/v*
_output_shapes
:
*
dtype0

Adam/softmaxLayer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*+
shared_nameAdam/softmaxLayer/kernel/v

.Adam/softmaxLayer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/softmaxLayer/kernel/v*
_output_shapes
:	
*
dtype0

Adam/reluLayer2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/reluLayer2/bias/v
~
*Adam/reluLayer2/bias/v/Read/ReadVariableOpReadVariableOpAdam/reluLayer2/bias/v*
_output_shapes	
:*
dtype0

Adam/reluLayer2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/reluLayer2/kernel/v

,Adam/reluLayer2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/reluLayer2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/reluLayer1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/reluLayer1/bias/v
~
*Adam/reluLayer1/bias/v/Read/ReadVariableOpReadVariableOpAdam/reluLayer1/bias/v*
_output_shapes	
:*
dtype0

Adam/reluLayer1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/reluLayer1/kernel/v

,Adam/reluLayer1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/reluLayer1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/softmaxLayer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/softmaxLayer/bias/m

,Adam/softmaxLayer/bias/m/Read/ReadVariableOpReadVariableOpAdam/softmaxLayer/bias/m*
_output_shapes
:
*
dtype0

Adam/softmaxLayer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*+
shared_nameAdam/softmaxLayer/kernel/m

.Adam/softmaxLayer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/softmaxLayer/kernel/m*
_output_shapes
:	
*
dtype0

Adam/reluLayer2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/reluLayer2/bias/m
~
*Adam/reluLayer2/bias/m/Read/ReadVariableOpReadVariableOpAdam/reluLayer2/bias/m*
_output_shapes	
:*
dtype0

Adam/reluLayer2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/reluLayer2/kernel/m

,Adam/reluLayer2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/reluLayer2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/reluLayer1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/reluLayer1/bias/m
~
*Adam/reluLayer1/bias/m/Read/ReadVariableOpReadVariableOpAdam/reluLayer1/bias/m*
_output_shapes	
:*
dtype0

Adam/reluLayer1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*)
shared_nameAdam/reluLayer1/kernel/m

,Adam/reluLayer1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/reluLayer1/kernel/m* 
_output_shapes
:
*
dtype0
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
z
softmaxLayer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namesoftmaxLayer/bias
s
%softmaxLayer/bias/Read/ReadVariableOpReadVariableOpsoftmaxLayer/bias*
_output_shapes
:
*
dtype0

softmaxLayer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*$
shared_namesoftmaxLayer/kernel
|
'softmaxLayer/kernel/Read/ReadVariableOpReadVariableOpsoftmaxLayer/kernel*
_output_shapes
:	
*
dtype0
w
reluLayer2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namereluLayer2/bias
p
#reluLayer2/bias/Read/ReadVariableOpReadVariableOpreluLayer2/bias*
_output_shapes	
:*
dtype0

reluLayer2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namereluLayer2/kernel
y
%reluLayer2/kernel/Read/ReadVariableOpReadVariableOpreluLayer2/kernel* 
_output_shapes
:
*
dtype0
w
reluLayer1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namereluLayer1/bias
p
#reluLayer1/bias/Read/ReadVariableOpReadVariableOpreluLayer1/bias*
_output_shapes	
:*
dtype0

reluLayer1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*"
shared_namereluLayer1/kernel
y
%reluLayer1/kernel/Read/ReadVariableOpReadVariableOpreluLayer1/kernel* 
_output_shapes
:
*
dtype0

serving_default_flatten_inputPlaceholder*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0* 
shape:ÿÿÿÿÿÿÿÿÿ
´
StatefulPartitionedCallStatefulPartitionedCallserving_default_flatten_inputreluLayer1/kernelreluLayer1/biasreluLayer2/kernelreluLayer2/biassoftmaxLayer/kernelsoftmaxLayer/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_392642

NoOpNoOp
µ3
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ð2
valueæ2Bã2 BÜ2
Î
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias*
¦
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias*
.
0
1
"2
#3
*4
+5*
.
0
1
"2
#3
*4
+5*
* 
°
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
6
1trace_0
2trace_1
3trace_2
4trace_3* 
6
5trace_0
6trace_1
7trace_2
8trace_3* 
* 
°
9iter

:beta_1

;beta_2
	<decay
=learning_ratemfmg"mh#mi*mj+mkvlvm"vn#vo*vp+vq*

>serving_default* 
* 
* 
* 

?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Dtrace_0* 

Etrace_0* 

0
1*

0
1*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Ktrace_0* 

Ltrace_0* 
a[
VARIABLE_VALUEreluLayer1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEreluLayer1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

"0
#1*

"0
#1*
* 

Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses*

Rtrace_0* 

Strace_0* 
a[
VARIABLE_VALUEreluLayer2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEreluLayer2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

*0
+1*

*0
+1*
* 

Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses*

Ytrace_0* 

Ztrace_0* 
c]
VARIABLE_VALUEsoftmaxLayer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEsoftmaxLayer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

[0
\1*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
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
8
]	variables
^	keras_api
	_total
	`count*
H
a	variables
b	keras_api
	ctotal
	dcount
e
_fn_kwargs*

_0
`1*

]	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

c0
d1*

a	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
~
VARIABLE_VALUEAdam/reluLayer1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/reluLayer1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/reluLayer2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/reluLayer2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/softmaxLayer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/softmaxLayer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/reluLayer1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/reluLayer1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/reluLayer2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/reluLayer2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUEAdam/softmaxLayer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/softmaxLayer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Û

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%reluLayer1/kernel/Read/ReadVariableOp#reluLayer1/bias/Read/ReadVariableOp%reluLayer2/kernel/Read/ReadVariableOp#reluLayer2/bias/Read/ReadVariableOp'softmaxLayer/kernel/Read/ReadVariableOp%softmaxLayer/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/reluLayer1/kernel/m/Read/ReadVariableOp*Adam/reluLayer1/bias/m/Read/ReadVariableOp,Adam/reluLayer2/kernel/m/Read/ReadVariableOp*Adam/reluLayer2/bias/m/Read/ReadVariableOp.Adam/softmaxLayer/kernel/m/Read/ReadVariableOp,Adam/softmaxLayer/bias/m/Read/ReadVariableOp,Adam/reluLayer1/kernel/v/Read/ReadVariableOp*Adam/reluLayer1/bias/v/Read/ReadVariableOp,Adam/reluLayer2/kernel/v/Read/ReadVariableOp*Adam/reluLayer2/bias/v/Read/ReadVariableOp.Adam/softmaxLayer/kernel/v/Read/ReadVariableOp,Adam/softmaxLayer/bias/v/Read/ReadVariableOpConst*(
Tin!
2	*
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
GPU 2J 8 *(
f#R!
__inference__traced_save_392905
º
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamereluLayer1/kernelreluLayer1/biasreluLayer2/kernelreluLayer2/biassoftmaxLayer/kernelsoftmaxLayer/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/reluLayer1/kernel/mAdam/reluLayer1/bias/mAdam/reluLayer2/kernel/mAdam/reluLayer2/bias/mAdam/softmaxLayer/kernel/mAdam/softmaxLayer/bias/mAdam/reluLayer1/kernel/vAdam/reluLayer1/bias/vAdam/reluLayer2/kernel/vAdam/reluLayer2/bias/vAdam/softmaxLayer/kernel/vAdam/softmaxLayer/bias/v*'
Tin 
2*
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
GPU 2J 8 *+
f&R$
"__inference__traced_restore_392996
©

ú
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392781

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í

-__inference_softmaxLayer_layer_call_fn_392790

inputs
unknown:	

	unknown_0:

identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392448o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

ú
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392414

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
½
_
C__inference_flatten_layer_call_and_return_conditional_losses_392741

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Í
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392455

inputs%
relulayer1_392415:
 
relulayer1_392417:	%
relulayer2_392432:
 
relulayer2_392434:	&
softmaxlayer_392449:	
!
softmaxlayer_392451:

identity¢"reluLayer1/StatefulPartitionedCall¢"reluLayer2/StatefulPartitionedCall¢$softmaxLayer/StatefulPartitionedCall·
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_392401
"reluLayer1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0relulayer1_392415relulayer1_392417*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392414
"reluLayer2/StatefulPartitionedCallStatefulPartitionedCall+reluLayer1/StatefulPartitionedCall:output:0relulayer2_392432relulayer2_392434*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392431¥
$softmaxLayer/StatefulPartitionedCallStatefulPartitionedCall+reluLayer2/StatefulPartitionedCall:output:0softmaxlayer_392449softmaxlayer_392451*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392448|
IdentityIdentity-softmaxLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
NoOpNoOp#^reluLayer1/StatefulPartitionedCall#^reluLayer2/StatefulPartitionedCall%^softmaxLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"reluLayer1/StatefulPartitionedCall"reluLayer1/StatefulPartitionedCall2H
"reluLayer2/StatefulPartitionedCall"reluLayer2/StatefulPartitionedCall2L
$softmaxLayer/StatefulPartitionedCall$softmaxLayer/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ	
¬
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392470
flatten_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392455o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
î
Ô
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392617
flatten_input%
relulayer1_392601:
 
relulayer1_392603:	%
relulayer2_392606:
 
relulayer2_392608:	&
softmaxlayer_392611:	
!
softmaxlayer_392613:

identity¢"reluLayer1/StatefulPartitionedCall¢"reluLayer2/StatefulPartitionedCall¢$softmaxLayer/StatefulPartitionedCall¾
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_392401
"reluLayer1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0relulayer1_392601relulayer1_392603*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392414
"reluLayer2/StatefulPartitionedCallStatefulPartitionedCall+reluLayer1/StatefulPartitionedCall:output:0relulayer2_392606relulayer2_392608*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392431¥
$softmaxLayer/StatefulPartitionedCallStatefulPartitionedCall+reluLayer2/StatefulPartitionedCall:output:0softmaxlayer_392611softmaxlayer_392613*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392448|
IdentityIdentity-softmaxLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
NoOpNoOp#^reluLayer1/StatefulPartitionedCall#^reluLayer2/StatefulPartitionedCall%^softmaxLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"reluLayer1/StatefulPartitionedCall"reluLayer1/StatefulPartitionedCall2H
"reluLayer2/StatefulPartitionedCall"reluLayer2/StatefulPartitionedCall2L
$softmaxLayer/StatefulPartitionedCall$softmaxLayer/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
Í

+__inference_reluLayer1_layer_call_fn_392750

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392414p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

ú
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392431

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥
D
(__inference_flatten_layer_call_fn_392735

inputs
identity¯
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_392401a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
î
Ô
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392597
flatten_input%
relulayer1_392581:
 
relulayer1_392583:	%
relulayer2_392586:
 
relulayer2_392588:	&
softmaxlayer_392591:	
!
softmaxlayer_392593:

identity¢"reluLayer1/StatefulPartitionedCall¢"reluLayer2/StatefulPartitionedCall¢$softmaxLayer/StatefulPartitionedCall¾
flatten/PartitionedCallPartitionedCallflatten_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_392401
"reluLayer1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0relulayer1_392581relulayer1_392583*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392414
"reluLayer2/StatefulPartitionedCallStatefulPartitionedCall+reluLayer1/StatefulPartitionedCall:output:0relulayer2_392586relulayer2_392588*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392431¥
$softmaxLayer/StatefulPartitionedCallStatefulPartitionedCall+reluLayer2/StatefulPartitionedCall:output:0softmaxlayer_392591softmaxlayer_392593*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392448|
IdentityIdentity-softmaxLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
NoOpNoOp#^reluLayer1/StatefulPartitionedCall#^reluLayer2/StatefulPartitionedCall%^softmaxLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"reluLayer1/StatefulPartitionedCall"reluLayer1/StatefulPartitionedCall2H
"reluLayer2/StatefulPartitionedCall"reluLayer2/StatefulPartitionedCall2L
$softmaxLayer/StatefulPartitionedCall$softmaxLayer/StatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
ä<
º
__inference__traced_save_392905
file_prefix0
,savev2_relulayer1_kernel_read_readvariableop.
*savev2_relulayer1_bias_read_readvariableop0
,savev2_relulayer2_kernel_read_readvariableop.
*savev2_relulayer2_bias_read_readvariableop2
.savev2_softmaxlayer_kernel_read_readvariableop0
,savev2_softmaxlayer_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_relulayer1_kernel_m_read_readvariableop5
1savev2_adam_relulayer1_bias_m_read_readvariableop7
3savev2_adam_relulayer2_kernel_m_read_readvariableop5
1savev2_adam_relulayer2_bias_m_read_readvariableop9
5savev2_adam_softmaxlayer_kernel_m_read_readvariableop7
3savev2_adam_softmaxlayer_bias_m_read_readvariableop7
3savev2_adam_relulayer1_kernel_v_read_readvariableop5
1savev2_adam_relulayer1_bias_v_read_readvariableop7
3savev2_adam_relulayer2_kernel_v_read_readvariableop5
1savev2_adam_relulayer2_bias_v_read_readvariableop9
5savev2_adam_softmaxlayer_kernel_v_read_readvariableop7
3savev2_adam_softmaxlayer_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
: õ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¥
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B ­
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_relulayer1_kernel_read_readvariableop*savev2_relulayer1_bias_read_readvariableop,savev2_relulayer2_kernel_read_readvariableop*savev2_relulayer2_bias_read_readvariableop.savev2_softmaxlayer_kernel_read_readvariableop,savev2_softmaxlayer_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_relulayer1_kernel_m_read_readvariableop1savev2_adam_relulayer1_bias_m_read_readvariableop3savev2_adam_relulayer2_kernel_m_read_readvariableop1savev2_adam_relulayer2_bias_m_read_readvariableop5savev2_adam_softmaxlayer_kernel_m_read_readvariableop3savev2_adam_softmaxlayer_bias_m_read_readvariableop3savev2_adam_relulayer1_kernel_v_read_readvariableop1savev2_adam_relulayer1_bias_v_read_readvariableop3savev2_adam_relulayer2_kernel_v_read_readvariableop1savev2_adam_relulayer2_bias_v_read_readvariableop5savev2_adam_softmaxlayer_kernel_v_read_readvariableop3savev2_adam_softmaxlayer_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 **
dtypes 
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Ð
_input_shapes¾
»: :
::
::	
:
: : : : : : : : : :
::
::	
:
:
::
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: 
¨

ú
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392448

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð3
Ì
!__inference__wrapped_model_392388
flatten_inputb
Nhandwritten_digit_recoginition_model_relulayer1_matmul_readvariableop_resource:
^
Ohandwritten_digit_recoginition_model_relulayer1_biasadd_readvariableop_resource:	b
Nhandwritten_digit_recoginition_model_relulayer2_matmul_readvariableop_resource:
^
Ohandwritten_digit_recoginition_model_relulayer2_biasadd_readvariableop_resource:	c
Phandwritten_digit_recoginition_model_softmaxlayer_matmul_readvariableop_resource:	
_
Qhandwritten_digit_recoginition_model_softmaxlayer_biasadd_readvariableop_resource:

identity¢Fhandwritten_digit_recoginition_model/reluLayer1/BiasAdd/ReadVariableOp¢Ehandwritten_digit_recoginition_model/reluLayer1/MatMul/ReadVariableOp¢Fhandwritten_digit_recoginition_model/reluLayer2/BiasAdd/ReadVariableOp¢Ehandwritten_digit_recoginition_model/reluLayer2/MatMul/ReadVariableOp¢Hhandwritten_digit_recoginition_model/softmaxLayer/BiasAdd/ReadVariableOp¢Ghandwritten_digit_recoginition_model/softmaxLayer/MatMul/ReadVariableOp
2handwritten_digit_recoginition_model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ¾
4handwritten_digit_recoginition_model/flatten/ReshapeReshapeflatten_input;handwritten_digit_recoginition_model/flatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
Ehandwritten_digit_recoginition_model/reluLayer1/MatMul/ReadVariableOpReadVariableOpNhandwritten_digit_recoginition_model_relulayer1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
6handwritten_digit_recoginition_model/reluLayer1/MatMulMatMul=handwritten_digit_recoginition_model/flatten/Reshape:output:0Mhandwritten_digit_recoginition_model/reluLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
Fhandwritten_digit_recoginition_model/reluLayer1/BiasAdd/ReadVariableOpReadVariableOpOhandwritten_digit_recoginition_model_relulayer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
7handwritten_digit_recoginition_model/reluLayer1/BiasAddBiasAdd@handwritten_digit_recoginition_model/reluLayer1/MatMul:product:0Nhandwritten_digit_recoginition_model/reluLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
4handwritten_digit_recoginition_model/reluLayer1/ReluRelu@handwritten_digit_recoginition_model/reluLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÖ
Ehandwritten_digit_recoginition_model/reluLayer2/MatMul/ReadVariableOpReadVariableOpNhandwritten_digit_recoginition_model_relulayer2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
6handwritten_digit_recoginition_model/reluLayer2/MatMulMatMulBhandwritten_digit_recoginition_model/reluLayer1/Relu:activations:0Mhandwritten_digit_recoginition_model/reluLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
Fhandwritten_digit_recoginition_model/reluLayer2/BiasAdd/ReadVariableOpReadVariableOpOhandwritten_digit_recoginition_model_relulayer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
7handwritten_digit_recoginition_model/reluLayer2/BiasAddBiasAdd@handwritten_digit_recoginition_model/reluLayer2/MatMul:product:0Nhandwritten_digit_recoginition_model/reluLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ±
4handwritten_digit_recoginition_model/reluLayer2/ReluRelu@handwritten_digit_recoginition_model/reluLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿÙ
Ghandwritten_digit_recoginition_model/softmaxLayer/MatMul/ReadVariableOpReadVariableOpPhandwritten_digit_recoginition_model_softmaxlayer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
8handwritten_digit_recoginition_model/softmaxLayer/MatMulMatMulBhandwritten_digit_recoginition_model/reluLayer2/Relu:activations:0Ohandwritten_digit_recoginition_model/softmaxLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Ö
Hhandwritten_digit_recoginition_model/softmaxLayer/BiasAdd/ReadVariableOpReadVariableOpQhandwritten_digit_recoginition_model_softmaxlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
9handwritten_digit_recoginition_model/softmaxLayer/BiasAddBiasAddBhandwritten_digit_recoginition_model/softmaxLayer/MatMul:product:0Phandwritten_digit_recoginition_model/softmaxLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
º
9handwritten_digit_recoginition_model/softmaxLayer/SoftmaxSoftmaxBhandwritten_digit_recoginition_model/softmaxLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

IdentityIdentityChandwritten_digit_recoginition_model/softmaxLayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
ý
NoOpNoOpG^handwritten_digit_recoginition_model/reluLayer1/BiasAdd/ReadVariableOpF^handwritten_digit_recoginition_model/reluLayer1/MatMul/ReadVariableOpG^handwritten_digit_recoginition_model/reluLayer2/BiasAdd/ReadVariableOpF^handwritten_digit_recoginition_model/reluLayer2/MatMul/ReadVariableOpI^handwritten_digit_recoginition_model/softmaxLayer/BiasAdd/ReadVariableOpH^handwritten_digit_recoginition_model/softmaxLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2
Fhandwritten_digit_recoginition_model/reluLayer1/BiasAdd/ReadVariableOpFhandwritten_digit_recoginition_model/reluLayer1/BiasAdd/ReadVariableOp2
Ehandwritten_digit_recoginition_model/reluLayer1/MatMul/ReadVariableOpEhandwritten_digit_recoginition_model/reluLayer1/MatMul/ReadVariableOp2
Fhandwritten_digit_recoginition_model/reluLayer2/BiasAdd/ReadVariableOpFhandwritten_digit_recoginition_model/reluLayer2/BiasAdd/ReadVariableOp2
Ehandwritten_digit_recoginition_model/reluLayer2/MatMul/ReadVariableOpEhandwritten_digit_recoginition_model/reluLayer2/MatMul/ReadVariableOp2
Hhandwritten_digit_recoginition_model/softmaxLayer/BiasAdd/ReadVariableOpHhandwritten_digit_recoginition_model/softmaxLayer/BiasAdd/ReadVariableOp2
Ghandwritten_digit_recoginition_model/softmaxLayer/MatMul/ReadVariableOpGhandwritten_digit_recoginition_model/softmaxLayer/MatMul/ReadVariableOp:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
Æ	
¬
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392577
flatten_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall°
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392545o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
±	
¥
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392676

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392545o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

$__inference_signature_wrapper_392642
flatten_input
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCallñ
StatefulPartitionedCallStatefulPartitionedCallflatten_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_392388o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Z V
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
'
_user_specified_nameflatten_input
½
_
C__inference_flatten_layer_call_and_return_conditional_losses_392401

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Í

+__inference_reluLayer2_layer_call_fn_392770

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÜ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392431p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ú
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392801

inputs1
matmul_readvariableop_resource:	
-
biasadd_readvariableop_resource:

identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ù
Í
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392545

inputs%
relulayer1_392529:
 
relulayer1_392531:	%
relulayer2_392534:
 
relulayer2_392536:	&
softmaxlayer_392539:	
!
softmaxlayer_392541:

identity¢"reluLayer1/StatefulPartitionedCall¢"reluLayer2/StatefulPartitionedCall¢$softmaxLayer/StatefulPartitionedCall·
flatten/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_flatten_layer_call_and_return_conditional_losses_392401
"reluLayer1/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0relulayer1_392529relulayer1_392531*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392414
"reluLayer2/StatefulPartitionedCallStatefulPartitionedCall+reluLayer1/StatefulPartitionedCall:output:0relulayer2_392534relulayer2_392536*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392431¥
$softmaxLayer/StatefulPartitionedCallStatefulPartitionedCall+reluLayer2/StatefulPartitionedCall:output:0softmaxlayer_392539softmaxlayer_392541*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392448|
IdentityIdentity-softmaxLayer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
·
NoOpNoOp#^reluLayer1/StatefulPartitionedCall#^reluLayer2/StatefulPartitionedCall%^softmaxLayer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2H
"reluLayer1/StatefulPartitionedCall"reluLayer1/StatefulPartitionedCall2H
"reluLayer2/StatefulPartitionedCall"reluLayer2/StatefulPartitionedCall2L
$softmaxLayer/StatefulPartitionedCall$softmaxLayer/StatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
È
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392703

inputs=
)relulayer1_matmul_readvariableop_resource:
9
*relulayer1_biasadd_readvariableop_resource:	=
)relulayer2_matmul_readvariableop_resource:
9
*relulayer2_biasadd_readvariableop_resource:	>
+softmaxlayer_matmul_readvariableop_resource:	
:
,softmaxlayer_biasadd_readvariableop_resource:

identity¢!reluLayer1/BiasAdd/ReadVariableOp¢ reluLayer1/MatMul/ReadVariableOp¢!reluLayer2/BiasAdd/ReadVariableOp¢ reluLayer2/MatMul/ReadVariableOp¢#softmaxLayer/BiasAdd/ReadVariableOp¢"softmaxLayer/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 reluLayer1/MatMul/ReadVariableOpReadVariableOp)relulayer1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
reluLayer1/MatMulMatMulflatten/Reshape:output:0(reluLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!reluLayer1/BiasAdd/ReadVariableOpReadVariableOp*relulayer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
reluLayer1/BiasAddBiasAddreluLayer1/MatMul:product:0)reluLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
reluLayer1/ReluRelureluLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 reluLayer2/MatMul/ReadVariableOpReadVariableOp)relulayer2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
reluLayer2/MatMulMatMulreluLayer1/Relu:activations:0(reluLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!reluLayer2/BiasAdd/ReadVariableOpReadVariableOp*relulayer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
reluLayer2/BiasAddBiasAddreluLayer2/MatMul:product:0)reluLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
reluLayer2/ReluRelureluLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"softmaxLayer/MatMul/ReadVariableOpReadVariableOp+softmaxlayer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
softmaxLayer/MatMulMatMulreluLayer2/Relu:activations:0*softmaxLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#softmaxLayer/BiasAdd/ReadVariableOpReadVariableOp,softmaxlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
softmaxLayer/BiasAddBiasAddsoftmaxLayer/MatMul:product:0+softmaxLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
softmaxLayer/SoftmaxSoftmaxsoftmaxLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
IdentityIdentitysoftmaxLayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp"^reluLayer1/BiasAdd/ReadVariableOp!^reluLayer1/MatMul/ReadVariableOp"^reluLayer2/BiasAdd/ReadVariableOp!^reluLayer2/MatMul/ReadVariableOp$^softmaxLayer/BiasAdd/ReadVariableOp#^softmaxLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!reluLayer1/BiasAdd/ReadVariableOp!reluLayer1/BiasAdd/ReadVariableOp2D
 reluLayer1/MatMul/ReadVariableOp reluLayer1/MatMul/ReadVariableOp2F
!reluLayer2/BiasAdd/ReadVariableOp!reluLayer2/BiasAdd/ReadVariableOp2D
 reluLayer2/MatMul/ReadVariableOp reluLayer2/MatMul/ReadVariableOp2J
#softmaxLayer/BiasAdd/ReadVariableOp#softmaxLayer/BiasAdd/ReadVariableOp2H
"softmaxLayer/MatMul/ReadVariableOp"softmaxLayer/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
±	
¥
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392659

inputs
unknown:

	unknown_0:	
	unknown_1:

	unknown_2:	
	unknown_3:	

	unknown_4:

identity¢StatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *i
fdRb
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392455o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ð
È
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392730

inputs=
)relulayer1_matmul_readvariableop_resource:
9
*relulayer1_biasadd_readvariableop_resource:	=
)relulayer2_matmul_readvariableop_resource:
9
*relulayer2_biasadd_readvariableop_resource:	>
+softmaxlayer_matmul_readvariableop_resource:	
:
,softmaxlayer_biasadd_readvariableop_resource:

identity¢!reluLayer1/BiasAdd/ReadVariableOp¢ reluLayer1/MatMul/ReadVariableOp¢!reluLayer2/BiasAdd/ReadVariableOp¢ reluLayer2/MatMul/ReadVariableOp¢#softmaxLayer/BiasAdd/ReadVariableOp¢"softmaxLayer/MatMul/ReadVariableOp^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ  m
flatten/ReshapeReshapeinputsflatten/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 reluLayer1/MatMul/ReadVariableOpReadVariableOp)relulayer1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
reluLayer1/MatMulMatMulflatten/Reshape:output:0(reluLayer1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!reluLayer1/BiasAdd/ReadVariableOpReadVariableOp*relulayer1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
reluLayer1/BiasAddBiasAddreluLayer1/MatMul:product:0)reluLayer1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
reluLayer1/ReluRelureluLayer1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 reluLayer2/MatMul/ReadVariableOpReadVariableOp)relulayer2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
reluLayer2/MatMulMatMulreluLayer1/Relu:activations:0(reluLayer2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!reluLayer2/BiasAdd/ReadVariableOpReadVariableOp*relulayer2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
reluLayer2/BiasAddBiasAddreluLayer2/MatMul:product:0)reluLayer2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿg
reluLayer2/ReluRelureluLayer2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"softmaxLayer/MatMul/ReadVariableOpReadVariableOp+softmaxlayer_matmul_readvariableop_resource*
_output_shapes
:	
*
dtype0
softmaxLayer/MatMulMatMulreluLayer2/Relu:activations:0*softmaxLayer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

#softmaxLayer/BiasAdd/ReadVariableOpReadVariableOp,softmaxlayer_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype0
softmaxLayer/BiasAddBiasAddsoftmaxLayer/MatMul:product:0+softmaxLayer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
p
softmaxLayer/SoftmaxSoftmaxsoftmaxLayer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
IdentityIdentitysoftmaxLayer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

NoOpNoOp"^reluLayer1/BiasAdd/ReadVariableOp!^reluLayer1/MatMul/ReadVariableOp"^reluLayer2/BiasAdd/ReadVariableOp!^reluLayer2/MatMul/ReadVariableOp$^softmaxLayer/BiasAdd/ReadVariableOp#^softmaxLayer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ: : : : : : 2F
!reluLayer1/BiasAdd/ReadVariableOp!reluLayer1/BiasAdd/ReadVariableOp2D
 reluLayer1/MatMul/ReadVariableOp reluLayer1/MatMul/ReadVariableOp2F
!reluLayer2/BiasAdd/ReadVariableOp!reluLayer2/BiasAdd/ReadVariableOp2D
 reluLayer2/MatMul/ReadVariableOp reluLayer2/MatMul/ReadVariableOp2J
#softmaxLayer/BiasAdd/ReadVariableOp#softmaxLayer/BiasAdd/ReadVariableOp2H
"softmaxLayer/MatMul/ReadVariableOp"softmaxLayer/MatMul/ReadVariableOp:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
©

ú
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392761

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¾m
·
"__inference__traced_restore_392996
file_prefix6
"assignvariableop_relulayer1_kernel:
1
"assignvariableop_1_relulayer1_bias:	8
$assignvariableop_2_relulayer2_kernel:
1
"assignvariableop_3_relulayer2_bias:	9
&assignvariableop_4_softmaxlayer_kernel:	
2
$assignvariableop_5_softmaxlayer_bias:
&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: %
assignvariableop_11_total_1: %
assignvariableop_12_count_1: #
assignvariableop_13_total: #
assignvariableop_14_count: @
,assignvariableop_15_adam_relulayer1_kernel_m:
9
*assignvariableop_16_adam_relulayer1_bias_m:	@
,assignvariableop_17_adam_relulayer2_kernel_m:
9
*assignvariableop_18_adam_relulayer2_bias_m:	A
.assignvariableop_19_adam_softmaxlayer_kernel_m:	
:
,assignvariableop_20_adam_softmaxlayer_bias_m:
@
,assignvariableop_21_adam_relulayer1_kernel_v:
9
*assignvariableop_22_adam_relulayer1_bias_v:	@
,assignvariableop_23_adam_relulayer2_kernel_v:
9
*assignvariableop_24_adam_relulayer2_bias_v:	A
.assignvariableop_25_adam_softmaxlayer_kernel_v:	
:
,assignvariableop_26_adam_softmaxlayer_bias_v:

identity_28¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9ø
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH¨
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B «
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_relulayer1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_relulayer1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_relulayer2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_relulayer2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp&assignvariableop_4_softmaxlayer_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp$assignvariableop_5_softmaxlayer_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_total_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_count_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOpassignvariableop_13_totalIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOpassignvariableop_14_countIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_relulayer1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_relulayer1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_relulayer2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_relulayer2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp.assignvariableop_19_adam_softmaxlayer_kernel_mIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp,assignvariableop_20_adam_softmaxlayer_bias_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_relulayer1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_relulayer1_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_relulayer2_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_relulayer2_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp.assignvariableop_25_adam_softmaxlayer_kernel_vIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_26AssignVariableOp,assignvariableop_26_adam_softmaxlayer_bias_vIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ¡
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_28Identity_28:output:0*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
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
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*¿
serving_default«
K
flatten_input:
serving_default_flatten_input:0ÿÿÿÿÿÿÿÿÿ@
softmaxLayer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿ
tensorflow/serving/predict:
è
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
¥
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
 __call__
*!&call_and_return_all_conditional_losses

"kernel
#bias"
_tf_keras_layer
»
$	variables
%trainable_variables
&regularization_losses
'	keras_api
(__call__
*)&call_and_return_all_conditional_losses

*kernel
+bias"
_tf_keras_layer
J
0
1
"2
#3
*4
+5"
trackable_list_wrapper
J
0
1
"2
#3
*4
+5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
,non_trainable_variables

-layers
.metrics
/layer_regularization_losses
0layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
É
1trace_0
2trace_1
3trace_2
4trace_32Þ
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392470
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392659
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392676
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392577¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z1trace_0z2trace_1z3trace_2z4trace_3
µ
5trace_0
6trace_1
7trace_2
8trace_32Ê
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392703
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392730
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392597
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392617¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z5trace_0z6trace_1z7trace_2z8trace_3
ÒBÏ
!__inference__wrapped_model_392388flatten_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¿
9iter

:beta_1

;beta_2
	<decay
=learning_ratemfmg"mh#mi*mj+mkvlvm"vn#vo*vp+vq"
	optimizer
,
>serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
?non_trainable_variables

@layers
Ametrics
Blayer_regularization_losses
Clayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ì
Dtrace_02Ï
(__inference_flatten_layer_call_fn_392735¢
²
FullArgSpec
args
jself
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
annotationsª *
 zDtrace_0

Etrace_02ê
C__inference_flatten_layer_call_and_return_conditional_losses_392741¢
²
FullArgSpec
args
jself
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
annotationsª *
 zEtrace_0
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ï
Ktrace_02Ò
+__inference_reluLayer1_layer_call_fn_392750¢
²
FullArgSpec
args
jself
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
annotationsª *
 zKtrace_0

Ltrace_02í
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392761¢
²
FullArgSpec
args
jself
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
annotationsª *
 zLtrace_0
%:#
2reluLayer1/kernel
:2reluLayer1/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Mnon_trainable_variables

Nlayers
Ometrics
Player_regularization_losses
Qlayer_metrics
	variables
trainable_variables
regularization_losses
 __call__
*!&call_and_return_all_conditional_losses
&!"call_and_return_conditional_losses"
_generic_user_object
ï
Rtrace_02Ò
+__inference_reluLayer2_layer_call_fn_392770¢
²
FullArgSpec
args
jself
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
annotationsª *
 zRtrace_0

Strace_02í
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392781¢
²
FullArgSpec
args
jself
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
annotationsª *
 zStrace_0
%:#
2reluLayer2/kernel
:2reluLayer2/bias
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Tnon_trainable_variables

Ulayers
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
$	variables
%trainable_variables
&regularization_losses
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
ñ
Ytrace_02Ô
-__inference_softmaxLayer_layer_call_fn_392790¢
²
FullArgSpec
args
jself
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
annotationsª *
 zYtrace_0

Ztrace_02ï
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392801¢
²
FullArgSpec
args
jself
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
annotationsª *
 zZtrace_0
&:$	
2softmaxLayer/kernel
:
2softmaxLayer/bias
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392470flatten_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392659inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392676inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392577flatten_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±B®
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392703inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
±B®
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392730inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392597flatten_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
¸Bµ
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392617flatten_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÑBÎ
$__inference_signature_wrapper_392642flatten_input"
²
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
annotationsª *
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
ÜBÙ
(__inference_flatten_layer_call_fn_392735inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
÷Bô
C__inference_flatten_layer_call_and_return_conditional_losses_392741inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
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
ßBÜ
+__inference_reluLayer1_layer_call_fn_392750inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
úB÷
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392761inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
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
ßBÜ
+__inference_reluLayer2_layer_call_fn_392770inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
úB÷
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392781inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
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
áBÞ
-__inference_softmaxLayer_layer_call_fn_392790inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
üBù
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392801inputs"¢
²
FullArgSpec
args
jself
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
annotationsª *
 
N
]	variables
^	keras_api
	_total
	`count"
_tf_keras_metric
^
a	variables
b	keras_api
	ctotal
	dcount
e
_fn_kwargs"
_tf_keras_metric
.
_0
`1"
trackable_list_wrapper
-
]	variables"
_generic_user_object
:  (2total
:  (2count
.
c0
d1"
trackable_list_wrapper
-
a	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
*:(
2Adam/reluLayer1/kernel/m
#:!2Adam/reluLayer1/bias/m
*:(
2Adam/reluLayer2/kernel/m
#:!2Adam/reluLayer2/bias/m
+:)	
2Adam/softmaxLayer/kernel/m
$:"
2Adam/softmaxLayer/bias/m
*:(
2Adam/reluLayer1/kernel/v
#:!2Adam/reluLayer1/bias/v
*:(
2Adam/reluLayer2/kernel/v
#:!2Adam/reluLayer2/bias/v
+:)	
2Adam/softmaxLayer/kernel/v
$:"
2Adam/softmaxLayer/bias/v§
!__inference__wrapped_model_392388"#*+:¢7
0¢-
+(
flatten_inputÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
softmaxLayer&#
softmaxLayerÿÿÿÿÿÿÿÿÿ
¤
C__inference_flatten_layer_call_and_return_conditional_losses_392741]3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
(__inference_flatten_layer_call_fn_392735P3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ×
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392597s"#*+B¢?
8¢5
+(
flatten_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ×
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392617s"#*+B¢?
8¢5
+(
flatten_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ð
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392703l"#*+;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 Ð
`__inference_handwritten_digit_recoginition_model_layer_call_and_return_conditional_losses_392730l"#*+;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 ¯
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392470f"#*+B¢?
8¢5
+(
flatten_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¯
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392577f"#*+B¢?
8¢5
+(
flatten_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
¨
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392659_"#*+;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
¨
E__inference_handwritten_digit_recoginition_model_layer_call_fn_392676_"#*+;¢8
1¢.
$!
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
¨
F__inference_reluLayer1_layer_call_and_return_conditional_losses_392761^0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_reluLayer1_layer_call_fn_392750Q0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
F__inference_reluLayer2_layer_call_and_return_conditional_losses_392781^"#0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_reluLayer2_layer_call_fn_392770Q"#0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ»
$__inference_signature_wrapper_392642"#*+K¢H
¢ 
Aª>
<
flatten_input+(
flatten_inputÿÿÿÿÿÿÿÿÿ";ª8
6
softmaxLayer&#
softmaxLayerÿÿÿÿÿÿÿÿÿ
©
H__inference_softmaxLayer_layer_call_and_return_conditional_losses_392801]*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ

 
-__inference_softmaxLayer_layer_call_fn_392790P*+0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
