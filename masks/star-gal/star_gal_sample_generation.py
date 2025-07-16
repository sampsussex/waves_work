from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.pylab as plt
import pyarrow.parquet as pq
import numpy as np
import pandas as pd
from tqdm.tqdm import tqdm
import matplotlib
import warnings
from sklearn import metrics


waves_n_filepath = 'WAVES-N_d1m3p1f1_Z22.parquet'
waves_s_filepath = 'WAVES-S_d1m3p1f1_Z22.parquet'
waves_n_stargal_filepath = 'WAVES-N_d1m3p1f1_Z22_stargal.parquet'
waves_s_stargal_filepath = 'WAVES-S_d1m3p1f1_Z22_stargal.parquet'

desi_edr_filepath = '../zall-tilecumulative-edr-vac.fits'
gama_filepath = '../gkvScienceCatv02.fits'
gama_star_filepath = '../StellarMassesGKVv24.fits'
gaia_s1_filepath = '../GAIA_S1.fits'
gaia_s2_filepath = '../GAIA_S2.fits'
gaia_n_filepath = '../GAIA_N.fits'
sdss_filepath = '../SDSS.csv'


# Load in star-gal classifications
df=pd.read_parquet(waves_n_stargal_filepath)
df2 = pd.read_parquet(waves_s_stargal_filepath)
df = pd.concat([df, df2], axis=0)
del df2

df['uberID'] = df['uberID'].astype(str)


# Load in parts of full catalog
read_columns = ['uberID', 'RAmax', 'Decmax', 'mask', 'starmask', 'ghostmask', 'class', 'duplicate', 'mag_Zt']

full_cat = pq.read_table(waves_n_filepath, columns=read_columns).to_pandas()
s_full_cat = pq.read_table(waves_s_filepath, columns=read_columns).to_pandas()
full_cat = pd.concat([full_cat, s_full_cat], axis=0)
del s_full_cat

# Filters I need to consider: duplicate == 0, class != 'artefact', mask == 0, starmask == 0, mag_Zt <= 21.25. Missing bands may come in later.
full_cat = full_cat[(full_cat['duplicate'] == 0) & 
                    (full_cat['class'] != 'artefact') & 
                    (full_cat['mask'] == 0) & 
                    (full_cat['starmask'] == 0) & 
                    (full_cat['mag_Zt'] <= 21.1)]

# Match parts of full catalog to star-gal classifications using pandas inner merge on UberID
full_cat = full_cat.merge(df, on='uberID', how='inner')
# Wrap RAs greater than 300 to negative 
full_cat.loc[full_cat[full_cat['RAmax'] > 300].index, 'RAmax'] = full_cat[full_cat['RAmax'] > 300]['RAmax'] - 360



# Load in DESI files
hdul = fits.open(desi_edr_filepath)
data = hdul[1].data
cols = hdul[1].columns
t=Table(data)
names = [name for name in t.colnames if len(t[name].shape) <= 1]
df_desi=t[names].to_pandas()
del hdul
del data
del cols
del t

df_desi=(df_desi[(df_desi['ZWARN']==0) & (df_desi['OBJTYPE']=='TGT')])

max_sep=0.6*u.arcsec
waves = SkyCoord(ra=full_cat['RAmax'].values*u.degree, dec=full_cat['Decmax'].values*u.degree)
desi = SkyCoord(ra=df_desi['TARGET_RA'].values*u.degree, dec=df_desi['TARGET_DEC'].values*u.degree)
idx, d2d, d3d = waves.match_to_catalog_sky(desi)
sep_constraint = d2d < max_sep
desi_matches=full_cat.iloc[sep_constraint]
desi_waves_matches=df_desi.iloc[idx[sep_constraint]]

desi_matches['spec_class'] = desi_waves_matches['SPECTYPE'].values
desi_matches['Z'] = desi_waves_matches['Z'].values
desi_matches['Zwarn'] = desi_waves_matches['ZWARN'].values
desi_matches['chi2'] = desi_waves_matches['CHI2'].values
desi_matches['RA_survey'] = desi_waves_matches['TARGET_RA'].values
desi_matches['Dec_survey'] = desi_waves_matches['TARGET_DEC'].values
desi_matches['morph'] = desi_waves_matches['MORPHTYPE'].values

desi_matches.loc[desi_matches[desi_matches['spec_class']=='GALAXY'].index,'spec_class']='galaxy'
desi_matches.loc[desi_matches[desi_matches['spec_class']=='STAR'].index,'spec_class']='star'
desi_matches.loc[desi_matches[desi_matches['spec_class']=='QSO'].index,'spec_class']='qso'

desi_matches.loc[desi_matches[desi_matches['Z']<0.0015].index,'spec_class']='star'

desi_matches=desi_matches.drop(desi_matches[(desi_matches['Z']<0.0015) & 
                                            (desi_matches['spec_class']=='galaxy')].index)

# Pring number of desi matches
print(f"Number of DESI matches: {len(desi_matches)}")

# Load in GAMA files

hdul = fits.open('../gkvScienceCatv02.fits')
data = hdul[1].data
cols = hdul[1].columns
t=Table(data)

df_gama=t.to_pandas().set_index('uberID')

df_gama=df_gama[df_gama['duplicate']==False]

df_gama=df_gama[df_gama['starmask']==False]
df_gama['mag_r_tot']=8.9-2.5*np.log10(df_gama['flux_rt'])

df_gama=df_gama[df_gama['NQ']>2]

df_gama.loc[df_gama[df_gama['RAmax']>300].index,'RAmax']=df_gama[df_gama['RAmax']>300]['RAmax']-360

hdul = fits.open('../StellarMassesGKVv24.fits')
data = hdul[1].data
cols = hdul[1].columns
t=Table(data)
names = [name for name in t.colnames if len(t[name].shape) <= 1]
gama_sm=t[names].to_pandas().set_index('uberID')
del hdul
del data
del cols
del t

df_gama['mstar']=np.nan
intersect=np.intersect1d(df_gama.index,gama_sm.index)
df_gama.loc[intersect,'mstar'] = gama_sm.loc[intersect]['mstar']

# Match GAMA to WAVES
max_sep=0.6*u.arcsec
waves = SkyCoord(ra=full_cat['RAmax'].values*u.degree, dec=full_cat['Decmax'].values*u.degree)
gama = SkyCoord(ra=df_gama['RAmax'].values*u.degree, dec=df_gama['Decmax'].values*u.degree)
idx, d2d, d3d = waves.match_to_catalog_sky(gama)
sep_constraint = d2d < max_sep
waves_matches=full_cat.iloc[sep_constraint]
gama_matches=df_gama.iloc[idx[sep_constraint]]



waves_matches['uberclass']=gama_matches['uberclass'].values
waves_matches['NQ']=gama_matches['NQ'].values
waves_matches['Z']=gama_matches['Z'].values
waves_matches['mstar']=gama_matches['mstar'].values
waves_matches['RA_survey']=gama_matches['RAmax'].values
waves_matches['Dec_survey']=gama_matches['Decmax'].values

print(f"Number of GAMA matches: {len(waves_matches)}")

waves_matches['spec_class']=waves_matches['uberclass']
waves_matches['spec_class']=waves_matches['spec_class'].replace([1], 'galaxy')
waves_matches['spec_class']=waves_matches['spec_class'].replace([2], 'star')
waves_matches['spec_class']=waves_matches['spec_class'].replace([3], 'ambiguous')

# Load in GAIA files 

hdul = fits.open('../GAIA_S1.fits')
data = hdul[1].data
cols = hdul[1].columns
t=Table(data)

dfgaia1=t.to_pandas()
del data
del cols
del t

hdul = fits.open('../GAIA_S2.fits')
data = hdul[1].data
cols = hdul[1].columns
t=Table(data)

dfgaia2=t.to_pandas()
del data
del cols
del t

hdul = fits.open('../GAIA_N.fits')
data = hdul[1].data
cols = hdul[1].columns
t=Table(data)

dfgaia3=t.to_pandas()
del data
del cols
del t

dfgaia=pd.concat([dfgaia1,dfgaia2,dfgaia3]).set_index('source_id')

dfgaia.loc[dfgaia[dfgaia['ra']>300].index,'ra'] = dfgaia[dfgaia['ra']>300]['ra']-360

max_sep=0.6*u.arcsec
waves = SkyCoord(ra=full_cat['RAmax'].values*u.degree, dec=full_cat['Decmax'].values*u.degree)
gaia = SkyCoord(ra=dfgaia['ra'].values*u.degree, dec=dfgaia['dec'].values*u.degree)
idx, d2d, d3d = waves.match_to_catalog_sky(gaia)
sep_constraint = d2d < max_sep
gaia_matches=full_cat.iloc[sep_constraint]
gaia_waves_matches = dfgaia.iloc[idx[sep_constraint]]

gaia_matches=pd.concat([gaia_matches.reset_index(),gaia_waves_matches.reset_index()],axis=1)

gaia_matches=gaia_matches.set_index('uberID')

gaia_matches['RA_survey']=gaia_waves_matches['ra'].values
gaia_matches['Dec_survey']=gaia_waves_matches['dec'].values

print(f"Number of GAIA matches: {len(gaia_matches)}")

# Load in SDSS files
df_sdss=pd.read_csv('../SDSS.csv')

max_sep=0.6*u.arcsec
waves = SkyCoord(ra=full_cat['RAmax'].values*u.degree, dec=full_cat['Decmax'].values*u.degree)
sdss = SkyCoord(ra=df_sdss['PLUG_RA'].values*u.degree, dec=df_sdss['PLUG_DEC'].values*u.degree)
idx, d2d, d3d = waves.match_to_catalog_sky(sdss)
sep_constraint = d2d < max_sep
sdss_matches=full_cat.iloc[sep_constraint]
sdss_waves_matches = df_sdss.iloc[idx[sep_constraint]]

sdss_matches['spec_class']=sdss_waves_matches['CLASS'].values
sdss_matches['Z']=sdss_waves_matches['Z'].values
sdss_matches['survey']=sdss_waves_matches['SURVEY'].values
sdss_matches['RA_survey']=sdss_waves_matches['PLUG_RA'].values
sdss_matches['Dec_survey']=sdss_waves_matches['PLUG_DEC'].values

sdss_matches.loc[sdss_matches[sdss_matches['spec_class']=='GALAXY'].index,'spec_class']='galaxy'
sdss_matches.loc[sdss_matches[sdss_matches['spec_class']=='STAR  '].index,'spec_class']='star'
sdss_matches.loc[sdss_matches[sdss_matches['spec_class']=='QSO   '].index,'spec_class']='qso'

waves_matches['gama_spec_class']=waves_matches['spec_class']
desi_matches['desi_spec_class']=desi_matches['spec_class']
sdss_matches['sdss_spec_class']=sdss_matches['spec_class']
gaia_matches['spec_class']='star'
gaia_matches['gaia_spec_class']=gaia_matches['spec_class']

stars=pd.concat([waves_matches[waves_matches['spec_class']=='star'],
                desi_matches[desi_matches['spec_class']=='star'],
                sdss_matches[sdss_matches['spec_class']=='star'],
                gaia_matches])


galaxies=pd.concat([waves_matches[waves_matches['spec_class']=='galaxy'],
                    desi_matches[desi_matches['spec_class']=='galaxy'],
                   sdss_matches[sdss_matches['spec_class']=='galaxy']])

total=pd.concat([stars,galaxies])

total.to_parquet('desiedr_gama_gaia_sdss_star_galaxy_sample.parquet', index=False)

desi_gals=desi_matches[desi_matches['spec_class']=='galaxy']
desi_stars=desi_matches[desi_matches['spec_class']=='star']
gama_gals=waves_matches[waves_matches['spec_class']=='galaxy']
gama_stars=waves_matches[waves_matches['spec_class']=='star']
sdss_gals=sdss_matches[sdss_matches['spec_class']=='galaxy']
sdss_stars=sdss_matches[sdss_matches['spec_class']=='star']

# massive plot
try:
    bins=np.linspace(14,21.2,50)

    fig = plt.figure(figsize=(20,12))
    spec = matplotlib.gridspec.GridSpec(ncols=4, nrows=2) # 6 columns evenly divides both 2 & 3
    plt.subplots_adjust(wspace=0.1)
    ax1 = fig.add_subplot(spec[0,0]) 
    ax2 = fig.add_subplot(spec[1,0])
    ax3 = fig.add_subplot(spec[1,3])
    ax4 = fig.add_subplot(spec[0,1]) 
    ax5 = fig.add_subplot(spec[1,1])
    ax6 = fig.add_subplot(spec[0,2]) 
    ax7 = fig.add_subplot(spec[1,2])

    ax1.hist(full_cat['mag_Zt'],histtype='stepfilled',bins=bins,color=(0.8,0.8,0.8),edgecolor='k',label='WAVES Total')
    ax1.hist(total['mag_Zt'],
            histtype='stepfilled',color=(0.5,0.5,0.5),bins=bins,edgecolor='k',label='Ground Truth Total')
    ax1.hist(waves_matches[waves_matches['spec_class']=='galaxy']['mag_Zt'],
                        histtype='stepfilled',color=(1,0.4,0.4),bins=bins,hatch='\\',edgecolor='k',label='GAMA galaxies')
    ax1.set_yscale('log')
    ax1.set_ylim(10,3000000)
    ax1.set_xlim(15,21.2)
    ax1.set_xticks([15,16,17,18,19,20,21])
    ax1.set_xlabel(r'$Z$-band magnitude')
    ax1.set_ylabel('Counts')
    ax1.set_title('GAMA galaxies \n'+str("{:,}".format(len(gama_gals))),fontsize=20,x=0.35,y=0.8)




    ax2.hist(full_cat['mag_Zt'],histtype='stepfilled',bins=bins,color=(0.8,0.8,0.8),edgecolor='k')
    ax2.hist(total['mag_Zt'],
            histtype='stepfilled',color=(0.5,0.5,0.5),bins=bins,edgecolor='k')
    ax2.hist(waves_matches[waves_matches['spec_class']=='star']['mag_Zt'],
                        histtype='stepfilled',color=(0.4,0.4,1),bins=bins,hatch='\\',edgecolor='k',label='GAMA stars')

    ax2.set_yscale('log')
    ax2.set_ylim(10,3000000)
    ax2.set_xlim(15,21.2)
    ax2.set_xticks([15,16,17,18,19,20,21])
    ax2.set_xlabel(r'$Z$-band magnitude')
    ax2.set_ylabel('Counts')
    ax2.set_title('GAMA stars \n'+str("{:,}".format(len(gama_stars))),fontsize=20,x=0.3,y=0.8)





    ax4.hist(full_cat['mag_Zt'],histtype='stepfilled',bins=bins,color=(0.8,0.8,0.8),edgecolor='k')
    ax4.hist(total['mag_Zt'],
            histtype='stepfilled',color=(0.5,0.5,0.5),bins=bins,edgecolor='k')
    ax4.hist(desi_matches[desi_matches['spec_class']=='galaxy']['mag_Zt'],
                        histtype='stepfilled',color=(1,0.4,0.4),bins=bins,hatch='+',edgecolor='k',label='DESI galaxies')

    ax4.set_yscale('log')
    ax4.set_ylim(10,3000000)
    ax4.set_xlim(15,21.2)
    ax4.set_xticks([15,16,17,18,19,20,21])
    ax4.set_xlabel(r'$Z$-band magnitude')
    ax4.set_xticks([15,16,17,18,19,20,21])
    ax4.set_yticklabels([])
    ax4.set_title('DESI galaxies \n'+str("{:,}".format(len(desi_gals))),fontsize=20,x=0.35,y=0.8)




    ax5.hist(full_cat['mag_Zt'],histtype='stepfilled',bins=bins,color=(0.8,0.8,0.8),edgecolor='k')
    ax5.hist(total['mag_Zt'],
            histtype='stepfilled',color=(0.5,0.5,0.5),bins=bins,edgecolor='k')
    ax5.hist(desi_matches[desi_matches['spec_class']=='star']['mag_Zt'],
                        histtype='stepfilled',color=(0.4,0.4,1),bins=bins,hatch='+',edgecolor='k',label='DESI stars')

    ax5.set_yscale('log')
    ax5.set_ylim(10,3000000)
    ax5.set_xticks([15,16,17,18,19,20,21])
    ax5.set_xlim(15,21.2)
    ax5.set_yticklabels([])
    ax5.set_xlabel(r'$Z$-band magnitude')
    ax5.set_title('DESI stars \n'+str("{:,}".format(len(desi_stars))),fontsize=20,x=0.3,y=0.8)


    ax3.hist(full_cat['mag_Zt'],histtype='stepfilled',bins=bins,color=(0.8,0.8,0.8),edgecolor='k')
    ax3.hist(total['mag_Zt'],
            histtype='stepfilled',color=(0.5,0.5,0.5),bins=bins,edgecolor='k')
    ax3.set_yscale('log')
    ax3.hist(gaia_matches['mag_Zt'],
                        histtype='stepfilled',color=(0.4,0.4,1),bins=bins,hatch='o',edgecolor='k',label='Gaia stars')

    ax3.set_yticklabels([])
    ax3.set_xlim(15,21.2)
    ax3.set_ylim(10,3000000)
    ax3.set_xticks([15,16,17,18,19,20,21])
    ax3.set_xlabel(r'$Z$-band magnitude')
    ax3.set_title('Gaia stars \n'+str("{:,}".format(len(gaia_matches))),fontsize=20,x=0.3,y=0.8)




    ax6.hist(full_cat['mag_Zt'],histtype='stepfilled',bins=bins,color=(0.8,0.8,0.8),edgecolor='k')
    ax6.hist(total['mag_Zt'],
            histtype='stepfilled',color=(0.5,0.5,0.5),bins=bins,edgecolor='k')
    ax6.hist(sdss_matches[sdss_matches['spec_class']=='galaxy']['mag_Zt'],
                        histtype='stepfilled',color=(1,0.4,0.4),bins=bins,hatch='x',edgecolor='k',label='SDSS galaxies')

    ax6.set_yscale('log')
    ax6.set_ylim(10,3000000)
    ax6.set_xlim(15,21.2)
    ax6.set_xticks([15,16,17,18,19,20,21])
    ax6.set_xlabel(r'$Z$-band magnitude')
    ax6.set_yticklabels([])
    ax6.set_title('SDSS galaxies \n'+str("{:,}".format(len(sdss_gals))),fontsize=20,x=0.35,y=0.8)




    ax7.hist(full_cat['mag_Zt'],histtype='stepfilled',bins=bins,color=(0.8,0.8,0.8),edgecolor='k')
    ax7.hist(total['mag_Zt'],
            histtype='stepfilled',color=(0.5,0.5,0.5),bins=bins,edgecolor='k')
    ax7.hist(sdss_matches[sdss_matches['spec_class']=='star']['mag_Zt'],
                        histtype='stepfilled',color=(0.4,0.4,1),bins=bins,hatch='x',edgecolor='k',label='SDSS stars')

    ax7.set_yscale('log')
    ax7.set_ylim(10,3000000)
    ax7.set_xlim(15,21.2)
    ax7.set_yticklabels([])
    ax7.set_xticks([15,16,17,18,19,20,21])
    ax7.set_xlabel(r'$Z$-band magnitude')
    ax7.set_title('SDSS stars \n'+str("{:,}".format(len(sdss_stars))),fontsize=20,x=0.3,y=0.8)

    fig.legend(loc=(0.75,0.6),frameon=False)

    plt.savefig('z_mag_count2.jpg',bbox_inches='tight',dpi=200)
    plt.clf()

except:
    print('Plotting failed, skipping...')

df_total=pd.concat([gaia_matches,desi_gals,desi_stars,
                    gama_gals,gama_stars,sdss_gals,sdss_stars])


# drop duplicates and contradictions
contradict=np.intersect1d(df_total[df_total['spec_class']=='star'].index,sdss_gals.index)
print(len(contradict))
duplicate=np.intersect1d(df_total[df_total['spec_class']=='galaxy'].index,sdss_gals.index)
print(len(duplicate))
df_total=pd.concat([df_total.drop(contradict).drop(duplicate),sdss_gals.drop(contradict)])

contradict=np.intersect1d(df_total[df_total['spec_class']=='galaxy'].index,sdss_stars.index)
print(len(contradict))
duplicate=np.intersect1d(df_total[df_total['spec_class']=='star'].index,sdss_stars.index)
print(len(duplicate))
df_total=pd.concat([df_total.drop(contradict).drop(duplicate),sdss_stars.drop(contradict)])

contradict=np.intersect1d(df_total[df_total['spec_class']=='star'].index,gama_gals.index)
print(len(contradict))
duplicate=np.intersect1d(df_total[df_total['spec_class']=='galaxy'].index,gama_gals.index)
print(len(duplicate))
df_total=pd.concat([df_total.drop(contradict).drop(duplicate),gama_gals.drop(contradict)])

contradict=np.intersect1d(df_total[df_total['spec_class']=='galaxy'].index,gama_stars.index)
print(len(contradict))
duplicate=np.intersect1d(df_total[df_total['spec_class']=='star'].index,gama_stars.index)
print(len(duplicate))
df_total=pd.concat([df_total.drop(contradict).drop(duplicate),gama_stars.drop(contradict)])

contradict=np.intersect1d(df_total[df_total['spec_class']=='star'].index,desi_gals.index)
print(len(contradict))
duplicate=np.intersect1d(df_total[df_total['spec_class']=='galaxy'].index,desi_gals.index)
print(len(duplicate))
df_total=pd.concat([df_total.drop(contradict).drop(duplicate),desi_gals.drop(contradict)])

contradict=np.intersect1d(df_total[df_total['spec_class']=='galaxy'].index,desi_stars.index)
print(len(contradict))
duplicate=np.intersect1d(df_total[df_total['spec_class']=='star'].index,desi_stars.index)
print(len(duplicate))
df_total=pd.concat([df_total.drop(contradict).drop(duplicate),desi_stars.drop(contradict)])


# save df total
df_total.to_parquet('desiedr_gama_gaia_sdss_star_galaxy_sample.parquet', index=False)


# F1 plot

confusion_matrix = metrics.confusion_matrix(df_total['spec_class'], 
                                            df_total['stargal'])

fig, ax = plt.subplots(1,1,figsize=(9,7))
data=confusion_matrix[[1,2]]
data2=np.zeros([2,3])
data2[0,0] = data[0,0]/sum(data[0])*100
data2[0,1] = data[0,1]/sum(data[0])*100
data2[0,2] = data[0,2]/sum(data[0])*100
data2[1,0] = data[1,0]/sum(data[1])*100
data2[1,1] = data[1,1]/sum(data[1])*100
data2[1,2] = data[1,2]/sum(data[1])*100

im=ax.imshow(data2/100,vmin=0,vmax=1,cmap='Blues')

x_label_list = ['Ambiguous','Galaxy', 'Star']
y_label_list = ['Galaxy','Star']

for i in range(0, data.shape[1]):
    for j in range(0, data.shape[0]):
        c = data[j,i]
        if c < sum(data[j])/2:
            ax.text(i, j-0.05, str(c), va='center', ha='center',color='black')
        else:
            ax.text(i, j-0.05, str(c), va='center', ha='center',color='white')
            
for i in range(0, data2.shape[1]):
    for j in range(0, data2.shape[0]):
        c = round(data2[j,i],2)
        if c < sum(data2[j])/2:
            ax.text(i, j+0.05, '('+str(c)+'%)', va='center', ha='center',color='black')
        else:
            ax.text(i, j+0.05, '('+str(c)+ '%)', va='center', ha='center',color='white')
            
fig.colorbar(im,label='Fraction of true uberclass',fraction=0.031, pad=0.04)
        
plt.xticks([0,1,2], x_label_list)
plt.yticks([0,1], y_label_list)

plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,
    left=False,
    right=False# ticks along the top edge are off
)

plt.xlabel('stargal label')
plt.ylabel('True label')

plt.savefig('WAVES_classifier_confusion_matrix_all.jpg',bbox_inches='tight',dpi=100,facecolor='white')

plt.clf()
