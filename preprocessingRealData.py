import sys
import io
import os
import allel
import numpy as np
import gzip
import pandas as pd
from sklearn.neighbors import NearestNeighbors

def pad(l, num, width):
    l.extend([num] * (width - len(l)))
    return l

def sort_min_diff(amat):
    mb = NearestNeighbors(len(amat), metric='manhattan').fit(amat)
    v = mb.kneighbors(amat)
    smallest = np.argmin(v[0].sum(axis=1))
    return amat[v[1][smallest]]

def reading_1000G_relationshipFile():
    print('READING 1000 GENOMES RELATIONSHIP FILE')
    relationShip_file = pd.read_csv('/home/noor/faststorage/popGen/realData/1000Genomes.csv',sep = '\t')
    print('READ SUCCESSFULLY!!!')
    print('\n')
    
    print('EXTRACTING AFRICAN AMERICANS ONLY')
    relationShip_file = relationShip_file[relationShip_file['Population'] == 'ASW']
    print('EXTRACTED SUCCESSFULLY')
    
    print('REMOVING CHILDREN')
    relationShip_file = relationShip_file[relationShip_file.Relationship != 'child']
    print('REMOVED SUCCESSFULLY!!!')
    
    return relationShip_file

def readMaskDataForScan(maskFileName, chrArm):
    isAccessible = []
    readingMasks = False
    if maskFileName.endswith(".gz"):
        fopen = gzip.open
    else:
        fopen = open
    with fopen(maskFileName, 'rt') as maskFile:
        for line in maskFile:
            if line.startswith(">"):
                
                currChr = line[1:].strip()
                #print('currchr : ',currChr)
                if currChr.startswith(chrArm):
                    #print('chrArm : ',chrArm)
                    readingMasks = True
                elif readingMasks:
                    break
            else:
                if readingMasks:
                    for char in line.strip().upper():
                        #print('Char : ',char)
                        if char == 'N':
                            isAccessible.append(False)
                        else:
                            isAccessible.append(True)
    return isAccessible

def polarizeSnps(unmasked, positions, refAlleles, altAlleles, ancArm):
    assert len(unmasked) == len(ancArm)
    assert len(positions) == len(refAlleles)
    assert len(positions) == len(altAlleles)
    isSnp = {}
    for i in range(len(positions)):
        isSnp[positions[i]] = i

    mapping = []
    for i in range(len(ancArm)):
        if ancArm[i] in 'ACGT':
            if i+1 in isSnp:
                ref, alt = refAlleles[isSnp[i+1]], altAlleles[isSnp[i+1]]
                if ancArm[i] == ref:
                    mapping.append([0, 1])  # no swap
                elif ancArm[i] == alt:
                    mapping.append([1, 0])  # swap
                else:
                    mapping.append([0, 1])  # no swap -- failed to polarize
                    unmasked[i] = False
        elif ancArm[i] == "N":
            unmasked[i] = False
            if i+1 in isSnp:
                mapping.append([0, 1])  # no swap -- failed to polarize
        else:
            sys.exit(
                "Found a character in ancestral chromosome "\
                 "that is not 'A', 'C', 'G', 'T' or 'N' (all upper case)!\n")
    assert len(mapping) == len(positions)
    return mapping, unmasked

def getSubWinBounds(chrLen, subWinSize):
    lastSubWinEnd = chrLen - chrLen % subWinSize
    lastSubWinStart = lastSubWinEnd - subWinSize + 1
    subWinBounds = []
    for subWinStart in range(1, lastSubWinStart+1, subWinSize):
        subWinEnd = subWinStart + subWinSize - 1
        subWinBounds.append((subWinStart, subWinEnd))
    return subWinBounds

def getSnpIndicesInSubWins(subWinSize, lastSubWinEnd, snpLocs):
    position_msOut = [[]]
    subWinStart = 1
    subWinEnd = subWinStart + subWinSize - 1
    snpIndicesInSubWins = [[]]
    for i in range(len(snpLocs)):
        while snpLocs[i] <= lastSubWinEnd and not (snpLocs[i] >= subWinStart and snpLocs[i] <= subWinEnd):
            subWinStart += subWinSize
            subWinEnd += subWinSize
            snpIndicesInSubWins.append([])
            position_msOut.append([])
        if snpLocs[i] <= lastSubWinEnd:
            snpIndicesInSubWins[-1].append(i)
            position_msOut[-1].append(snpLocs[i])
    while subWinEnd < lastSubWinEnd:
        snpIndicesInSubWins.append([])
        position_msOut.append([])
        subWinStart += subWinSize
        subWinEnd += subWinSize
    return snpIndicesInSubWins,position_msOut

if __name__ == "__main__":
    outFilePath, chrArm = sys.argv[1:]
    
    chrom_type = 'chr' + str(chrArm)
    
    print('CALLING FUNCTION TO EXTRACT AFR-AMR ONLY')
    relationShip_file = reading_1000G_relationshipFile()
    samples = list(relationShip_file['Sample'])
    print('SAMPLES COLLECTED SUCCESSFULLY!!!')
    
    print('READING VCF FILE FOR ',chrom_type)
    completePath_vcf = '/home/noor/faststorage/popGen/realData/repeatMasker/filtered_withoutRepeats_' + chrom_type + '.vcf.gz' 
    chrArmFile = allel.read_vcf(completePath_vcf)
    print('VCF FILE READ SUCCESSFULLY')
    
    chroms = chrArmFile["variants/CHROM"]
    positions = np.extract(chroms == chrArm, chrArmFile["variants/POS"])
    
    print('READING FASTA FILE')
    completePath_fasta = '/home/noor/faststorage/popGen/realData/pilotMask/20160622.' + chrom_type + '.pilot_mask.fasta.gz'
    unmasked = readMaskDataForScan(completePath_fasta, chrom_type)
    print('FILE READ SUCCESSFULLY!!!')
    
    chrLen = len(unmasked)
    winSize = 1100000
    numSubWins = 11
    subWinSize = int(winSize/numSubWins)
    unmaskedFracCutoff = 0.25
    print('sub window size : ',subWinSize)
    
    sampleIndicesToKeep = [i for i in range(len(samples))]
    rawgenos = np.take(chrArmFile["calldata/GT"], [i for i in range(len(chroms)) if chroms[i] == chrArm], axis=0)
    genos = allel.GenotypeArray(rawgenos)
    
    refAlleles = np.extract(chroms == chrArm, chrArmFile['variants/REF'])
    altAlleles = np.extract(chroms == chrArm, chrArmFile['variants/ALT'])
    
    genos = allel.GenotypeArray(genos.subset(sel1=sampleIndicesToKeep))
    alleleCounts = genos.count_alleles()
    
    #remove all but mono/biallelic unmasked sites
    isBiallelic = alleleCounts.is_biallelic()
    for i in range(len(isBiallelic)):
        if not isBiallelic[i]:
            unmasked[positions[i]-1] = False
            
    snpIndicesToKeep = [i for i in range(len(positions)) if unmasked[positions[i]-1]]
    
    genos = allel.GenotypeArray(genos.subset(sel0=snpIndicesToKeep))
    positions = [positions[i] for i in snpIndicesToKeep]
    alleleCounts = allel.AlleleCountsArray([[alleleCounts[i][0], max(alleleCounts[i][1:])] for i in snpIndicesToKeep])
    
    haps = genos.to_haplotypes()
    
    subWinBounds = getSubWinBounds(chrLen, subWinSize)
    
    goodSubWins = []
    lastSubWinEnd = chrLen - chrLen % subWinSize
    
    snpIndicesInSubWins,position_msOut = getSnpIndicesInSubWins(subWinSize, lastSubWinEnd, positions)
    
    subWinIndex = 0
    lastSubWinStart = lastSubWinEnd - subWinSize + 1
    
    
    lst_pos = []
    counter = 0
    for subWinStart in range(1, lastSubWinStart+1, subWinSize):
        subWinEnd = subWinStart + subWinSize - 1
        unmaskedFrac = unmasked[subWinStart-1:subWinEnd].count(True)/float(subWinEnd-subWinStart+1)

        print('unmasked fraction : ',unmaskedFrac)

        print('len : ',len(snpIndicesInSubWins[subWinIndex]))
        if len(snpIndicesInSubWins[subWinIndex]) > 0 and unmaskedFrac >= unmaskedFracCutoff:
            lst = []
            print('inside if ')
            print('subWinStart : ',subWinStart, 'subWinEnd : ',subWinEnd)
            #lst_pos.append([])
            #for i in range(len(position_msOut[subWinIndex])):
            print(type(position_msOut[subWinIndex]))
            lst_pos.append(position_msOut[subWinIndex])

            print(len(position_msOut[subWinIndex]))
            hapsInSubWin = allel.HaplotypeArray(haps.subset(sel0=snpIndicesInSubWins[subWinIndex]))
            lst.append(hapsInSubWin)
            
            l = (np.asarray(lst[0])).T
            res = np.asarray(sort_min_diff(np.array([pad(list(x),0,5000) for x in l],dtype=int)).T)
            print('shape : ',res.shape)
            
            completePath_output = outFilePath + '/' + chrom_type + '/' + chrom_type + '_' + str(subWinStart) + '_' + str(subWinEnd)
            np.save(completePath_output, np.asarray(res))
            
            #counter += 1
        print('\n')
        subWinIndex += 1
        
        
        
        
        
        
        
        