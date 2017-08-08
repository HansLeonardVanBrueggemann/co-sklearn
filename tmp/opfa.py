# =========================================================================
#  Copyright Het Nederlands Kanker Instituut - Antoni van Leeuwenhoek
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0.txt
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ========================================================================

import numpy
import scipy.stats
import scipy.linalg

def orthogonalPrincipalFeatureSelection(X):

    D, N = numpy.shape(X)

    selected = []
    sse = []

    for i in range(0, 100):

        # Compute Largest Eigenvector
        
        C = numpy.dot(X.T, X)
        v, w = numpy.linalg.eigh(C)
        target = w[:,N-1]
        
        # Compute most correlated feature
        
        f = [ scipy.stats.pearsonr(X[d,:], target)[0] for d in range(0, D) ]
        f = numpy.array(f)
        f = numpy.argmax(f)
        
        selected.append(f)
        
        # Compute projection matrix
        
        F = X[f,:][numpy.newaxis].T
        P = numpy.eye(N) - (numpy.divide(numpy.dot(F, F.T), numpy.dot(F.T, F)))
        
        # Project
        
        X = numpy.dot(X,P)
        
        # Compute SSE
        
        s = scipy.linalg.svdvals(numpy.delete(X, selected, 0).T)
        sse.append(numpy.max(s))
        
    return selected, sse